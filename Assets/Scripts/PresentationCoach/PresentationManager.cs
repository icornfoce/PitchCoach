using System;
using System.Threading.Tasks;
using UnityEngine;

namespace PresentationCoach
{
    public enum SessionState { Idle, Recording, Analyzing, Complete, Error }

    /// <summary>
    /// The session brain. Starts/stops capture, accumulates live time, assembles the
    /// final <see cref="SessionData"/> from the Audio + Vision managers, and requests
    /// LLM feedback when the user finishes. UI listens to the events here and to the
    /// sub-managers (exposed below) for real-time meters.
    /// </summary>
    public class PresentationManager : MonoBehaviour
    {
        [Header("Subsystems")]
        [SerializeField] private AudioSpeechManager audioManager;
        [SerializeField] private VisionManager visionManager;
        [SerializeField] private LLMAdvisorAPI advisor;

        [Header("Session")]
        [Tooltip("Auto-stop after this many seconds (0 = manual stop only).")]
        [SerializeField] private float maxDurationSeconds = 0f;

        // ---- Exposed for the UI to subscribe to live signals ----
        public AudioSpeechManager Audio => audioManager;
        public VisionManager Vision => visionManager;

        public SessionState State { get; private set; } = SessionState.Idle;
        public float ElapsedSeconds { get; private set; }
        public SessionData LatestSession { get; private set; }

        // ---- Lifecycle events ----
        public event Action<SessionState> OnStateChanged;
        public event Action<SessionData, string> OnSessionComplete;  // data + feedback text
        public event Action<string> OnError;

        private bool _busy;   // guards overlapping start/stop calls

        // ============================================================== //
        //  Start / Stop (async, button-friendly wrappers)                //
        // ============================================================== //

        /// <summary>Hook this to the "Start" UI button (async void is fine for events).</summary>
        public async void StartSession()
        {
            if (State == SessionState.Recording || _busy) return;
            _busy = true;
            try
            {
                ElapsedSeconds = 0f;
                LatestSession = null;
                visionManager.BeginSession();
                await audioManager.StartSessionAsync();   // mic + Azure spin-up
                SetState(SessionState.Recording);
            }
            catch (Exception e)
            {
                Fail($"Failed to start session: {e.Message}");
            }
            finally { _busy = false; }
        }

        /// <summary>Hook this to the "Stop" UI button. Stops capture, then asks the LLM.</summary>
        public async void StopSession()
        {
            if (State != SessionState.Recording || _busy) return;
            _busy = true;
            try
            {
                await audioManager.StopSessionAsync();
                visionManager.EndSession();

                LatestSession = BuildSessionData();
                SetState(SessionState.Analyzing);

                string feedback = await RequestFeedbackAsync(LatestSession);

                SetState(SessionState.Complete);
                OnSessionComplete?.Invoke(LatestSession, feedback);
            }
            catch (Exception e)
            {
                Fail($"Failed to finalize session: {e.Message}");
            }
            finally { _busy = false; }
        }

        // ============================================================== //
        //  Per-frame                                                     //
        // ============================================================== //

        private void Update()
        {
            if (State != SessionState.Recording) return;

            ElapsedSeconds += Time.deltaTime;
            if (maxDurationSeconds > 0f && ElapsedSeconds >= maxDurationSeconds)
                StopSession();
        }

        // ============================================================== //
        //  Aggregation                                                   //
        // ============================================================== //

        /// <summary>Snapshots both managers' running aggregates into a SessionData.</summary>
        private SessionData BuildSessionData()
        {
            var data = new SessionData
            {
                totalTimeSeconds      = ElapsedSeconds,
                totalWords            = audioManager.TotalWords,
                fillerWordCount       = audioManager.FillerCount,
                averageVolumeDb       = audioManager.AverageVolumeDb,
                averagePitchHz        = audioManager.AveragePitchHz,
                eyeContactPercentage  = visionManager.EyeContactPercentage,
                openPosturePercentage = visionManager.OpenPosturePercentage,
            };
            data.RecomputeDerived();   // fills wordsPerMinute + postureStatus verdict
            return data;
        }

        /// <summary>Calls the advisor; returns a graceful fallback string on failure.</summary>
        private async Task<string> RequestFeedbackAsync(SessionData data)
        {
            if (advisor == null) return "AI advisor not configured.";
            try
            {
                return await advisor.GetFeedbackAsync(data);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[Presentation] LLM feedback failed: {e.Message}");
                // Don't fail the whole session just because the network hiccuped —
                // the user still gets their scored radar chart.
                return "ไม่สามารถเชื่อมต่อโค้ช AI ได้ในขณะนี้ แต่คะแนนของคุณถูกบันทึกแล้ว";
            }
        }

        // ============================================================== //
        //  State helpers                                                 //
        // ============================================================== //

        private void SetState(SessionState s)
        {
            State = s;
            OnStateChanged?.Invoke(s);
        }

        private void Fail(string message)
        {
            Debug.LogError($"[Presentation] {message}");
            SetState(SessionState.Error);
            OnError?.Invoke(message);
        }
    }
}
