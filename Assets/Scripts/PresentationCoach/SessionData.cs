using System;
using System.Collections.Generic;
using UnityEngine;

namespace PresentationCoach
{
    /// <summary>
    /// Coarse posture classification derived from upper-body landmarks.
    /// "Open" = shoulders square and wide to camera (confident);
    /// "Closed" = shoulders hunched/narrow or arms crossed (defensive).
    /// </summary>
    public enum PostureStatus
    {
        Unknown = 0,
        Open    = 1,
        Closed  = 2
    }

    /// <summary>
    /// Serializable snapshot of a single practice session.
    /// Produced incrementally by PresentationManager and consumed by both
    /// LLMAdvisorAPI (as a JSON payload) and UIManager (for the radar chart).
    ///
    /// All "raw" fields are accumulated live; <see cref="RecomputeDerived"/>
    /// turns them into the human-readable metrics, and <see cref="GetRadarScores"/>
    /// maps everything onto a normalized 0..100 scale for visualization/feedback.
    /// </summary>
    [Serializable]
    public class SessionData
    {
        // ---- Timing -------------------------------------------------------
        [Tooltip("Total spoken/active session length in seconds.")]
        public float totalTimeSeconds;

        // ---- Speech content ----------------------------------------------
        [Tooltip("Total recognized words across all final STT results.")]
        public int totalWords;

        [Tooltip("Derived: words per minute = totalWords / (totalTime/60).")]
        public float wordsPerMinute;

        [Tooltip("Count of filler words/sounds (e.g. Thai 'เอ่อ', 'อืม').")]
        public int fillerWordCount;

        // ---- Audio DSP ----------------------------------------------------
        [Tooltip("Mean loudness over the session in dBFS (negative; 0 = full scale).")]
        public float averageVolumeDb = -60f;

        [Tooltip("Mean fundamental pitch in Hz (0 if never voiced).")]
        public float averagePitchHz;

        // ---- Vision -------------------------------------------------------
        [Range(0f, 100f)]
        [Tooltip("Percent of frames the user's head was oriented toward camera.")]
        public float eyeContactPercentage;

        [Tooltip("Dominant posture across the session.")]
        public PostureStatus postureStatus = PostureStatus.Unknown;

        [Range(0f, 100f)]
        [Tooltip("Percent of frames classified as 'Open' posture.")]
        public float openPosturePercentage;

        // -------------------------------------------------------------------
        // Tuning constants. Public so they can be exposed in the inspector or
        // localized; these define what "ideal" looks like for the scoring curves.
        // -------------------------------------------------------------------
        public const float IdealWpmMin = 110f;   // below this = too slow
        public const float IdealWpmMax = 150f;   // above this = too fast
        public const float IdealWpmHardLow = 70f;
        public const float IdealWpmHardHigh = 200f;

        public const float TargetVolumeDb = -18f; // comfortable presentation loudness
        public const float MinVolumeDb = -45f;     // effectively inaudible/mumbling

        // A "clean" talk has roughly < 1 filler every 30s. We score against that rate.
        public const float FillersPerMinutePenaltyFull = 6f; // >=6/min => 0 points

        /// <summary>
        /// Recomputes the human-readable derived metrics from the raw accumulators.
        /// Call once when the session ends (or any time you need a consistent view).
        /// </summary>
        public void RecomputeDerived()
        {
            // WPM: guard against divide-by-zero for very short/empty sessions.
            float minutes = Mathf.Max(totalTimeSeconds, 0.001f) / 60f;
            wordsPerMinute = totalWords / minutes;

            // Posture verdict is just whichever class dominated the timeline.
            postureStatus = openPosturePercentage >= 50f ? PostureStatus.Open : PostureStatus.Closed;
        }

        /// <summary>
        /// Order of axes on the radar chart. Keep in sync with <see cref="GetRadarScores"/>.
        /// </summary>
        public static readonly string[] RadarAxes =
        {
            "Pace",        // speaking speed in the ideal WPM band
            "Volume",      // loudness near the target dB
            "Fluency",     // inverse of filler-word rate
            "Eye Contact", // % time facing camera
            "Posture"      // % time open posture
        };

        /// <summary>
        /// Maps the session onto five 0..100 axis scores for the XCharts radar.
        /// Each sub-score uses a documented curve so the LLM and the chart agree.
        /// </summary>
        public List<float> GetRadarScores()
        {
            return new List<float>
            {
                ScorePace(wordsPerMinute),
                ScoreVolume(averageVolumeDb),
                ScoreFluency(),
                Mathf.Clamp(eyeContactPercentage, 0f, 100f),
                Mathf.Clamp(openPosturePercentage, 0f, 100f)
            };
        }

        /// <summary>
        /// Pace score: 100 inside the ideal band, falling linearly to 0 at the
        /// hard low/high bounds. A trapezoid centered on [IdealWpmMin, IdealWpmMax].
        /// </summary>
        public static float ScorePace(float wpm)
        {
            if (wpm >= IdealWpmMin && wpm <= IdealWpmMax) return 100f;
            if (wpm < IdealWpmMin)
                return 100f * Mathf.InverseLerp(IdealWpmHardLow, IdealWpmMin, wpm);
            return 100f * Mathf.InverseLerp(IdealWpmHardHigh, IdealWpmMax, wpm);
        }

        /// <summary>
        /// Volume score: 100 at/above the target dB, fading to 0 at the mumble floor.
        /// (We don't penalize "too loud" here since clipping is handled upstream.)
        /// </summary>
        public static float ScoreVolume(float db)
        {
            return 100f * Mathf.Clamp01(Mathf.InverseLerp(MinVolumeDb, TargetVolumeDb, db));
        }

        /// <summary>
        /// Fluency score: penalize fillers by *rate* (per minute), not raw count,
        /// so long talks aren't unfairly punished. 0 fillers => 100, at/above the
        /// full-penalty rate => 0.
        /// </summary>
        public float ScoreFluency()
        {
            float minutes = Mathf.Max(totalTimeSeconds, 0.001f) / 60f;
            float fillersPerMin = fillerWordCount / minutes;
            return 100f * (1f - Mathf.Clamp01(fillersPerMin / FillersPerMinutePenaltyFull));
        }

        /// <summary>Overall single-number score = mean of the radar axes.</summary>
        public float OverallScore()
        {
            var scores = GetRadarScores();
            float sum = 0f;
            foreach (var s in scores) sum += s;
            return sum / scores.Count;
        }

        /// <summary>Serializes to JSON for the LLM payload / debugging / persistence.</summary>
        public string ToJson(bool pretty = false) => JsonUtility.ToJson(this, pretty);

        public static SessionData FromJson(string json) => JsonUtility.FromJson<SessionData>(json);
    }
}
