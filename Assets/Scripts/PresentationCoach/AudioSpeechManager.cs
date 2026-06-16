using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;

// Azure Cognitive Services Speech SDK (DLLs live under Assets/SpeechSDK).
// Only loaded when SttEngine.AzureStreaming is selected (needs Azure.Core.dll).
using Microsoft.CognitiveServices.Speech;
using Microsoft.CognitiveServices.Speech.Audio;

namespace PresentationCoach
{
    /// <summary>How spoken words are turned into text for scoring.</summary>
    public enum SttEngine
    {
        /// <summary>Offline fake words/fillers (no key, editor/dev).</summary>
        Simulated,
        /// <summary>Record the whole session, transcribe once at Stop via OpenAI Whisper (reuses the OpenAI key).</summary>
        OpenAIWhisper,
        /// <summary>Live continuous Azure recognition (requires Azure.Core.dll alongside the Speech SDK).</summary>
        AzureStreaming
    }

    /// <summary>
    /// Owns everything ear-related:
    ///   1) Native DSP — captures the mic with Unity's <see cref="Microphone"/> and
    ///      runs an FFT (<c>GetSpectrumData</c>) every frame to estimate live
    ///      loudness (dBFS) and fundamental pitch (Hz).
    ///   2) Speech-to-text — streams the same mic audio to Azure for continuous
    ///      recognition, counts words, and tallies filler sounds (เอ่อ / อืม / ...).
    ///
    /// Public events fire on the Unity main thread; Azure callbacks arrive on
    /// worker threads, so their results are marshaled via Interlocked counters
    /// and flushed in <see cref="Update"/>.
    /// </summary>
    public class AudioSpeechManager : MonoBehaviour
    {
        [Header("Microphone / DSP")]
        [SerializeField] private int sampleRate = 16000;        // 16 kHz = Azure's preferred PCM rate
        [SerializeField] private int spectrumSize = 1024;       // FFT bins (power of two)
        [SerializeField] private FFTWindow fftWindow = FFTWindow.BlackmanHarris;
        [Tooltip("RMS below this (dBFS) is treated as silence and ignored for averages.")]
        [SerializeField] private float silenceFloorDb = -50f;
        [Tooltip("Plays the captured mic clip so Unity's GetSpectrumData can analyze it. " +
                 "Route its Output to a SILENT AudioMixerGroup to avoid speaker feedback.")]
        [SerializeField] private AudioSource monitorSource;

        [Header("Speech-to-Text engine")]
        [Tooltip("OpenAIWhisper = transcribe the whole session at Stop (reuses your OpenAI key). " +
                 "Simulated = offline fake words. AzureStreaming = live Azure (needs Azure.Core.dll).")]
        [SerializeField] private SttEngine sttEngine = SttEngine.OpenAIWhisper;

        [Header("OpenAI Whisper")]
        [Tooltip("Same OpenAI key you use for the LLM advisor.")]
        [SerializeField] private string openAiApiKey = "<OPENAI_API_KEY>";
        [SerializeField] private string whisperEndpoint = "https://api.openai.com/v1/audio/transcriptions";
        [SerializeField] private string whisperModel = "whisper-1";
        [Tooltip("Safety cap on recorded seconds sent to Whisper (API limit is 25MB ≈ 13 min @16kHz).")]
        [SerializeField] private int maxRecordSeconds = 480;

        [Header("Azure Speech (optional; needs Azure.Core.dll)")]
        [SerializeField] private string azureKey = "<AZURE_SPEECH_KEY>";
        [SerializeField] private string azureRegion = "southeastasia";
        [Tooltip("Azure recognition language; first 2 letters also used as the Whisper language hint.")]
        [SerializeField] private string recognitionLanguage = "th-TH";

        [Header("Filler words")]
        [Tooltip("Tokens counted as fillers. Defaults to common Thai hesitation sounds.")]
        [SerializeField] private List<string> fillerWords = new List<string> { "เอ่อ", "อืม", "เอิ่ม", "อ่า", "เออ" };

        // ---- Live, frame-accurate readouts (consumed by UI/PresentationManager) ----
        public float CurrentVolumeDb { get; private set; } = -80f;
        public float CurrentPitchHz  { get; private set; }
        public bool  IsVoiced        { get; private set; }   // true when above silence floor

        // ---- Session aggregates (read when the session ends) ----
        public int TotalWords  { get; private set; }
        public int FillerCount { get; private set; }
        public float AverageVolumeDb => _volumeSampleCount > 0 ? _volumeDbSum / _volumeSampleCount : silenceFloorDb;
        public float AveragePitchHz  => _pitchSampleCount  > 0 ? _pitchHzSum  / _pitchSampleCount  : 0f;

        // ---- Events (main thread) ----
        public event Action<float> OnVolumeUpdated;          // dBFS
        public event Action<float> OnPitchUpdated;           // Hz
        public event Action<string> OnPartialTranscript;     // interim hypothesis
        public event Action<string> OnFinalTranscript;       // committed utterance
        public event Action<int> OnFillerDetected;           // running total

        // ---- Internals ----
        private AudioClip _micClip;
        private string _micDevice;
        private float[] _spectrum;
        private bool _capturing;
        private bool _monitorStarted;

        // Whisper: full-session PCM accumulated from the mic ring buffer.
        private List<float> _sessionSamples;
        private int _maxSamples;

        // Azure objects (null in simulation mode).
        private SpeechRecognizer _recognizer;
        private PushAudioInputStream _pushStream;
        private int _lastMicPosition;
        private const int MaxAzureChunk = 4096;   // max mic samples streamed to Azure per frame

        // Thread-safe handoff from Azure worker threads -> main thread.
        private int _pendingWords;
        private int _pendingFillers;
        private string _pendingFinal;
        private readonly object _textLock = new object();

        private float _volumeDbSum; private int _volumeSampleCount;
        private float _pitchHzSum;  private int _pitchSampleCount;

        // ============================================================== //
        //  Lifecycle                                                     //
        // ============================================================== //

        /// <summary>Begins mic capture and (optionally) Azure continuous recognition.</summary>
        public async Task StartSessionAsync()
        {
            _spectrum = new float[spectrumSize];
            ResetAggregates();

            StartMicrophone();
            _capturing = true;

            if (sttEngine == SttEngine.AzureStreaming)
                await StartAzureAsync();
        }

        /// <summary>Stops recognition and mic capture; safe to call multiple times.</summary>
        public async Task StopSessionAsync()
        {
            _capturing = false;

            if (sttEngine == SttEngine.AzureStreaming && _recognizer != null)
            {
                try { await _recognizer.StopContinuousRecognitionAsync(); }
                catch (Exception e) { Debug.LogWarning($"[Audio] Azure stop failed: {e.Message}"); }
                _recognizer.Dispose(); _recognizer = null;
                _pushStream?.Close(); _pushStream?.Dispose(); _pushStream = null;
            }

            // Grab the final audio fragment before closing the mic.
            if (sttEngine == SttEngine.OpenAIWhisper) AppendSessionAudio();

            if (_micDevice != null && Microphone.IsRecording(_micDevice))
                Microphone.End(_micDevice);

            // Whisper transcribes the whole session in one request (awaited so
            // TotalWords/FillerCount are ready before PresentationManager reads them).
            if (sttEngine == SttEngine.OpenAIWhisper)
                await TranscribeWithWhisperAsync();
        }

        private void ResetAggregates()
        {
            TotalWords = FillerCount = 0;
            _volumeDbSum = _pitchHzSum = 0f;
            _volumeSampleCount = _pitchSampleCount = 0;
            _lastMicPosition = 0;

            if (_sessionSamples == null) _sessionSamples = new List<float>();
            _sessionSamples.Clear();
            _maxSamples = Mathf.Max(1, maxRecordSeconds) * sampleRate;
        }

        // ============================================================== //
        //  Native DSP: Microphone + FFT                                  //
        // ============================================================== //

        private void StartMicrophone()
        {
            if (Microphone.devices.Length == 0)
            {
                Debug.LogError("[Audio] No microphone devices found.");
                return;
            }
            _micDevice = Microphone.devices[0];
            // Looping 1-second buffer; we read it both for the FFT and for Azure.
            _micClip = Microphone.Start(_micDevice, true, 1, sampleRate);

            // Route the mic into an AudioSource so we can run GetSpectrumData on it.
            // Playback is deferred to Update() until the ring buffer has real samples.
            if (monitorSource == null) monitorSource = gameObject.AddComponent<AudioSource>();
            monitorSource.clip = _micClip;
            monitorSource.loop = true;
            _monitorStarted = false;
        }

        private void Update()
        {
            if (!_capturing || _micClip == null) return;

            // Defer monitor playback until the mic has buffered samples, else the
            // first frames would analyze silence.
            if (!_monitorStarted && Microphone.GetPosition(_micDevice) > 0)
            { monitorSource.Play(); _monitorStarted = true; }

            AnalyzeSpectrum();              // volume + pitch every frame
            switch (sttEngine)
            {
                case SttEngine.AzureStreaming: PumpAudioToAzure();   break;  // stream to Azure
                case SttEngine.OpenAIWhisper:  AppendSessionAudio(); break;  // buffer for end-of-session upload
                case SttEngine.Simulated:      SimulateStt();        break;  // fake words
            }

            FlushRecognitionResults();      // surface Azure/simulated worker-thread results
        }

        /// <summary>
        /// Reads the mic into a spectrum buffer and derives loudness + pitch.
        ///
        /// Loudness (dBFS): RMS of the magnitude spectrum, converted to decibels:
        ///     rms = sqrt(Σ binÂ²)
        ///     dB  = 20 · log10(rms)        (0 dBFS = full scale, values are negative)
        ///
        /// Pitch (Hz): index of the dominant magnitude bin mapped back to frequency.
        ///     binWidth = (sampleRate / 2) / spectrumSize   // Nyquist spread over bins
        ///     f0       = peakIndex · binWidth
        /// This is a cheap "dominant frequency" estimate — fine for a coaching meter,
        /// not a substitute for a true autocorrelation/YIN pitch tracker.
        /// </summary>
        private void AnalyzeSpectrum()
        {
            if (monitorSource == null || !monitorSource.isPlaying) return;
            // Per-source FFT isolates the mic from the rest of the scene's audio.
            monitorSource.GetSpectrumData(_spectrum, 0, fftWindow);

            float sumSq = 0f;
            int peakIndex = 0; float peakMag = 0f;
            for (int i = 1; i < _spectrum.Length; i++)
            {
                float m = _spectrum[i];
                sumSq += m * m;
                if (m > peakMag) { peakMag = m; peakIndex = i; }
            }

            float rms = Mathf.Sqrt(sumSq / _spectrum.Length);
            float db  = rms > 1e-7f ? 20f * Mathf.Log10(rms) : -80f;

            float binWidth = (sampleRate * 0.5f) / _spectrum.Length;
            float pitch    = peakIndex * binWidth;

            CurrentVolumeDb = db;
            CurrentPitchHz  = pitch;
            IsVoiced        = db > silenceFloorDb;

            OnVolumeUpdated?.Invoke(db);
            OnPitchUpdated?.Invoke(pitch);

            // Only fold *voiced* frames into the session averages.
            if (IsVoiced)
            {
                _volumeDbSum += db; _volumeSampleCount++;
                if (pitch > 50f && pitch < 500f)   // plausible human vocal range
                { _pitchHzSum += pitch; _pitchSampleCount++; }
            }
        }

        // ============================================================== //
        //  Azure STT                                                     //
        // ============================================================== //

        private async Task StartAzureAsync()
        {
            try
            {
                var config = SpeechConfig.FromSubscription(azureKey, azureRegion);
                config.SpeechRecognitionLanguage = recognitionLanguage;

                // Push raw 16-bit/16kHz/mono PCM pulled from the Unity mic each frame.
                var fmt = AudioStreamFormat.GetWaveFormatPCM((uint)sampleRate, 16, 1);
                _pushStream = AudioInputStream.CreatePushStream(fmt);
                var audioConfig = AudioConfig.FromStreamInput(_pushStream);

                _recognizer = new SpeechRecognizer(config, audioConfig);

                // Interim hypotheses -> live caption (no counting yet).
                _recognizer.Recognizing += (s, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Result.Text))
                        lock (_textLock) { _pendingPartial = e.Result.Text; }
                };

                // Final utterances -> count words + fillers (worker thread!).
                _recognizer.Recognized += (s, e) =>
                {
                    if (e.Result.Reason != ResultReason.RecognizedSpeech) return;
                    string text = e.Result.Text;
                    if (string.IsNullOrWhiteSpace(text)) return;

                    int words   = CountWords(text);
                    int fillers = CountFillers(text);

                    Interlocked.Add(ref _pendingWords, words);
                    Interlocked.Add(ref _pendingFillers, fillers);
                    lock (_textLock) { _pendingFinal = text; }
                };

                _recognizer.Canceled += (s, e) =>
                    Debug.LogWarning($"[Audio] Azure canceled: {e.Reason} / {e.ErrorDetails}");

                await _recognizer.StartContinuousRecognitionAsync();
            }
            catch (Exception e)
            {
                Debug.LogError($"[Audio] Azure init failed, falling back to simulation: {e.Message}");
                sttEngine = SttEngine.Simulated;
            }
        }

        /// <summary>
        /// Copies any new mic samples since last frame into the Azure push stream,
        /// converting Unity's float [-1,1] PCM to signed 16-bit little-endian.
        /// </summary>
        private void PumpAudioToAzure()
        {
            if (_pushStream == null || _micClip == null) return;

            int pos = Microphone.GetPosition(_micDevice);
            int count = pos - _lastMicPosition;
            if (count < 0) count += _micClip.samples;   // wrapped around the ring buffer
            if (count <= 0) return;

            // Read at most to the end of the ring buffer this frame to avoid a GetData
            // overrun; the wrapped remainder is picked up on the next frame.
            int offset = _lastMicPosition % _micClip.samples;
            count = Mathf.Min(count, MaxAzureChunk);
            count = Mathf.Min(count, _micClip.samples - offset);
            if (count <= 0) return;

            var floatBuf = new float[count];
            _micClip.GetData(floatBuf, offset);

            var bytes = new byte[count * 2];
            for (int i = 0; i < count; i++)
            {
                short s = (short)(Mathf.Clamp(floatBuf[i], -1f, 1f) * short.MaxValue);
                bytes[i * 2]     = (byte)(s & 0xff);          // little-endian PCM16
                bytes[i * 2 + 1] = (byte)((s >> 8) & 0xff);
            }
            _pushStream.Write(bytes);
            _lastMicPosition = (offset + count) % _micClip.samples;
        }

        private string _pendingPartial;

        /// <summary>Moves worker-thread recognition results onto the Unity main thread.</summary>
        private void FlushRecognitionResults()
        {
            int w = Interlocked.Exchange(ref _pendingWords, 0);
            int f = Interlocked.Exchange(ref _pendingFillers, 0);
            if (w > 0) TotalWords += w;
            if (f > 0) { FillerCount += f; OnFillerDetected?.Invoke(FillerCount); }

            string partial, final;
            lock (_textLock) { partial = _pendingPartial; _pendingPartial = null; final = _pendingFinal; _pendingFinal = null; }
            if (!string.IsNullOrEmpty(partial)) OnPartialTranscript?.Invoke(partial);
            if (!string.IsNullOrEmpty(final))   OnFinalTranscript?.Invoke(final);
        }

        // ============================================================== //
        //  Text analysis helpers                                         //
        // ============================================================== //

        /// <summary>
        /// Word count. Thai is written without spaces, so for th-TH we approximate
        /// with character count / avg-word-length; for spaced languages we split.
        /// (Swap in a proper Thai word-breaker like ICU/PyThaiNLP for accuracy.)
        /// </summary>
        private int CountWords(string text)
        {
            if (recognitionLanguage.StartsWith("th"))
            {
                int chars = text.Replace(" ", "").Length;
                return Mathf.Max(1, Mathf.RoundToInt(chars / 3f)); // ~3 chars/word heuristic
            }
            return text.Split(new[] { ' ', '\t', '\n' }, StringSplitOptions.RemoveEmptyEntries).Length;
        }

        /// <summary>Counts occurrences of each configured filler token in an utterance.</summary>
        private int CountFillers(string text)
        {
            int total = 0;
            foreach (var filler in fillerWords)
            {
                if (string.IsNullOrEmpty(filler)) continue;
                int idx = 0;
                while ((idx = text.IndexOf(filler, idx, StringComparison.Ordinal)) != -1)
                { total++; idx += filler.Length; }
            }
            return total;
        }

        // ============================================================== //
        //  OpenAI Whisper STT (record session -> transcribe at Stop)     //
        // ============================================================== //

        /// <summary>Appends new mic samples (since last read) into the session buffer.</summary>
        private void AppendSessionAudio()
        {
            if (_micClip == null || _sessionSamples == null) return;

            int pos = Microphone.GetPosition(_micDevice);
            int count = pos - _lastMicPosition;
            if (count < 0) count += _micClip.samples;   // ring buffer wrapped
            if (count <= 0) return;

            // Read in pieces bounded by the end of the ring buffer.
            while (count > 0 && _sessionSamples.Count < _maxSamples)
            {
                int offset = _lastMicPosition % _micClip.samples;
                int chunk = Mathf.Min(count, _micClip.samples - offset);
                var buf = new float[chunk];
                _micClip.GetData(buf, offset);
                _sessionSamples.AddRange(buf);
                _lastMicPosition = (offset + chunk) % _micClip.samples;
                count -= chunk;
            }
            // Even if we hit the cap, keep the read cursor moving so we don't re-read.
            if (count > 0) _lastMicPosition = pos;
        }

        /// <summary>
        /// Encodes the captured session to WAV and sends it to OpenAI Whisper, then
        /// counts words + fillers from the returned transcript.
        /// </summary>
        private async Task TranscribeWithWhisperAsync()
        {
            if (_sessionSamples == null || _sessionSamples.Count == 0)
            {
                Debug.LogWarning("[Audio] No audio captured; skipping Whisper transcription.");
                return;
            }
            if (string.IsNullOrEmpty(openAiApiKey) || openAiApiKey.StartsWith("<"))
            {
                Debug.LogWarning("[Audio] OpenAI API key not set; skipping Whisper transcription.");
                return;
            }

            byte[] wav = EncodeWav(_sessionSamples, sampleRate);
            string lang = (recognitionLanguage != null && recognitionLanguage.Length >= 2)
                ? recognitionLanguage.Substring(0, 2).ToLowerInvariant() : "th";

            var form = new List<IMultipartFormSection>
            {
                new MultipartFormFileSection("file", wav, "session.wav", "audio/wav"),
                new MultipartFormDataSection("model", whisperModel),
                new MultipartFormDataSection("language", lang),
                new MultipartFormDataSection("response_format", "json"),
            };

            using (var req = UnityWebRequest.Post(whisperEndpoint, form))
            {
                req.SetRequestHeader("Authorization", "Bearer " + openAiApiKey);
                await req.SendWebRequest();   // awaitable via the extension in LLMAdvisorAPI

                if (req.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogWarning($"[Audio] Whisper request failed [{req.responseCode}]: {req.error}\n{req.downloadHandler.text}");
                    return;
                }

                var parsed = JsonUtility.FromJson<WhisperResponse>(req.downloadHandler.text);
                string text = parsed != null ? parsed.text : null;
                if (string.IsNullOrWhiteSpace(text))
                {
                    Debug.LogWarning("[Audio] Whisper returned empty text.");
                    return;
                }

                TotalWords = CountWords(text);
                FillerCount = CountFillers(text);
                OnFinalTranscript?.Invoke(text);
                OnFillerDetected?.Invoke(FillerCount);
            }
        }

        [Serializable] private class WhisperResponse { public string text; }

        /// <summary>Writes mono 16-bit PCM samples as a little-endian WAV (RIFF) byte array.</summary>
        private static byte[] EncodeWav(List<float> samples, int sampleRate)
        {
            int n = samples.Count;
            using (var ms = new System.IO.MemoryStream(44 + n * 2))
            using (var w = new System.IO.BinaryWriter(ms))
            {
                int byteRate = sampleRate * 2;          // mono * 16-bit
                w.Write(System.Text.Encoding.ASCII.GetBytes("RIFF"));
                w.Write(36 + n * 2);                    // chunk size
                w.Write(System.Text.Encoding.ASCII.GetBytes("WAVE"));
                w.Write(System.Text.Encoding.ASCII.GetBytes("fmt "));
                w.Write(16);                            // subchunk1 size (PCM)
                w.Write((short)1);                      // audio format = PCM
                w.Write((short)1);                      // channels = mono
                w.Write(sampleRate);
                w.Write(byteRate);
                w.Write((short)2);                      // block align
                w.Write((short)16);                     // bits per sample
                w.Write(System.Text.Encoding.ASCII.GetBytes("data"));
                w.Write(n * 2);                         // data size
                for (int i = 0; i < n; i++)
                {
                    short s = (short)(Mathf.Clamp(samples[i], -1f, 1f) * short.MaxValue);
                    w.Write(s);
                }
                w.Flush();
                return ms.ToArray();
            }
        }

        // ============================================================== //
        //  Simulation (offline / no credentials)                         //
        // ============================================================== //

        private float _simTimer;

        /// <summary>
        /// Fake STT: every ~2s emits a couple of words and occasionally a filler,
        /// but only while the mic is actually voiced — so the demo still reacts
        /// to the user's real loudness.
        /// </summary>
        private void SimulateStt()
        {
            _simTimer += Time.deltaTime;
            if (_simTimer < 2f || !IsVoiced) return;
            _simTimer = 0f;

            int words = UnityEngine.Random.Range(4, 9);
            Interlocked.Add(ref _pendingWords, words);
            if (UnityEngine.Random.value < 0.35f)
            {
                Interlocked.Add(ref _pendingFillers, 1);
                lock (_textLock) { _pendingFinal = fillerWords.Count > 0 ? fillerWords[0] + " ..." : "..."; }
            }
        }

        private void OnDestroy()
        {
            if (_micDevice != null && Microphone.IsRecording(_micDevice)) Microphone.End(_micDevice);
            _recognizer?.Dispose();
            _pushStream?.Dispose();
        }
    }
}
