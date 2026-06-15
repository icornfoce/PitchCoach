using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

// Azure Cognitive Services Speech SDK (DLLs live under Assets/SpeechSDK).
// Wrapped in a try/catch + simulation toggle so the project still runs in the
// editor without valid credentials.
using Microsoft.CognitiveServices.Speech;
using Microsoft.CognitiveServices.Speech.Audio;

namespace PresentationCoach
{
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

        [Header("Azure Speech")]
        [SerializeField] private string azureKey = "<AZURE_SPEECH_KEY>";
        [SerializeField] private string azureRegion = "southeastasia";
        [Tooltip("Recognition language. th-TH for Thai filler detection.")]
        [SerializeField] private string recognitionLanguage = "th-TH";
        [Tooltip("If true, skip Azure and emit fake words/fillers (editor/offline dev).")]
        [SerializeField] private bool useSimulatedStt = false;

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

            if (!useSimulatedStt)
                await StartAzureAsync();
        }

        /// <summary>Stops recognition and mic capture; safe to call multiple times.</summary>
        public async Task StopSessionAsync()
        {
            _capturing = false;

            if (_recognizer != null)
            {
                try { await _recognizer.StopContinuousRecognitionAsync(); }
                catch (Exception e) { Debug.LogWarning($"[Audio] Azure stop failed: {e.Message}"); }
                _recognizer.Dispose(); _recognizer = null;
                _pushStream?.Close(); _pushStream?.Dispose(); _pushStream = null;
            }

            if (_micDevice != null && Microphone.IsRecording(_micDevice))
                Microphone.End(_micDevice);
        }

        private void ResetAggregates()
        {
            TotalWords = FillerCount = 0;
            _volumeDbSum = _pitchHzSum = 0f;
            _volumeSampleCount = _pitchSampleCount = 0;
            _lastMicPosition = 0;
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
            if (!useSimulatedStt) PumpAudioToAzure();
            else SimulateStt();

            FlushRecognitionResults();      // surface Azure worker-thread results
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
                useSimulatedStt = true;
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
