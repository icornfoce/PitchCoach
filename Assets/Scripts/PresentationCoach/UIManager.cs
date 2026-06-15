using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using XCharts.Runtime;

namespace PresentationCoach
{
    /// <summary>
    /// Drives the two screens:
    ///   • Practice panel — live volume bar, pitch/timer/filler readouts, eye-contact
    ///     dot and posture label, fed by the Audio/Vision manager events.
    ///   • Summary panel  — XCharts radar of the five scores plus the LLM feedback.
    ///
    /// This class is purely presentational: it subscribes to events and writes to
    /// widgets. All logic/state lives in PresentationManager and the sub-managers.
    /// </summary>
    public class UIManager : MonoBehaviour
    {
        [Header("Core")]
        [SerializeField] private PresentationManager manager;

        [Header("Panels")]
        [SerializeField] private GameObject practicePanel;
        [SerializeField] private GameObject analyzingPanel;
        [SerializeField] private GameObject summaryPanel;

        [Header("Practice — live widgets")]
        [SerializeField] private Image volumeBar;            // fillAmount 0..1
        [SerializeField] private TMP_Text pitchText;
        [SerializeField] private TMP_Text timerText;
        [SerializeField] private TMP_Text fillerText;
        [SerializeField] private TMP_Text transcriptText;
        [SerializeField] private Image eyeContactDot;        // tinted green/grey
        [SerializeField] private TMP_Text postureText;

        [Header("Practice — volume mapping (dBFS -> bar)")]
        [SerializeField] private float dbMin = -50f;
        [SerializeField] private float dbMax = -10f;

        [Header("Summary")]
        [SerializeField] private RadarChart radarChart;
        [SerializeField] private TMP_Text feedbackText;
        [SerializeField] private TMP_Text overallScoreText;
        [SerializeField] private TMP_Text statsText;

        [Header("Colors")]
        [SerializeField] private Color activeColor = new Color(0.30f, 0.80f, 0.45f);
        [SerializeField] private Color idleColor   = new Color(0.55f, 0.55f, 0.55f);

        // ============================================================== //
        //  Wiring                                                        //
        // ============================================================== //

        private void Start()
        {
            // Lifecycle from the controller.
            manager.OnStateChanged   += HandleStateChanged;
            manager.OnSessionComplete += HandleSessionComplete;
            manager.OnError          += HandleError;

            // Live signals straight from the sub-managers (lowest latency).
            manager.Audio.OnVolumeUpdated    += HandleVolume;
            manager.Audio.OnPitchUpdated     += HandlePitch;
            manager.Audio.OnFillerDetected   += HandleFiller;
            manager.Audio.OnPartialTranscript += HandleTranscript;
            manager.Audio.OnFinalTranscript  += HandleTranscript;
            manager.Vision.OnEyeContactChanged += HandleEyeContact;
            manager.Vision.OnPostureChanged    += HandlePosture;

            ShowOnly(null);   // hide all until a session starts
        }

        private void OnDestroy()
        {
            if (manager == null) return;
            manager.OnStateChanged   -= HandleStateChanged;
            manager.OnSessionComplete -= HandleSessionComplete;
            manager.OnError          -= HandleError;
            if (manager.Audio != null)
            {
                manager.Audio.OnVolumeUpdated    -= HandleVolume;
                manager.Audio.OnPitchUpdated     -= HandlePitch;
                manager.Audio.OnFillerDetected   -= HandleFiller;
                manager.Audio.OnPartialTranscript -= HandleTranscript;
                manager.Audio.OnFinalTranscript  -= HandleTranscript;
            }
            if (manager.Vision != null)
            {
                manager.Vision.OnEyeContactChanged -= HandleEyeContact;
                manager.Vision.OnPostureChanged    -= HandlePosture;
            }
        }

        private void Update()
        {
            // Timer is the one value we poll rather than receive via event.
            if (manager.State == SessionState.Recording && timerText != null)
            {
                float t = manager.ElapsedSeconds;
                timerText.text = $"{(int)t / 60:00}:{(int)t % 60:00}";
            }
        }

        // ============================================================== //
        //  Live practice handlers                                        //
        // ============================================================== //

        private void HandleVolume(float db)
        {
            if (volumeBar == null) return;
            volumeBar.fillAmount = Mathf.Clamp01(Mathf.InverseLerp(dbMin, dbMax, db));
        }

        private void HandlePitch(float hz)
        {
            if (pitchText != null) pitchText.text = hz > 50f ? $"{hz:F0} Hz" : "—";
        }

        private void HandleFiller(int total)
        {
            if (fillerText != null) fillerText.text = total.ToString();
        }

        private void HandleTranscript(string text)
        {
            if (transcriptText != null) transcriptText.text = text;
        }

        private void HandleEyeContact(bool contact)
        {
            if (eyeContactDot != null) eyeContactDot.color = contact ? activeColor : idleColor;
        }

        private void HandlePosture(PostureStatus status)
        {
            if (postureText == null) return;
            postureText.text = status == PostureStatus.Open ? "Open" : "Closed";
            postureText.color = status == PostureStatus.Open ? activeColor : idleColor;
        }

        // ============================================================== //
        //  Panel switching                                               //
        // ============================================================== //

        private void HandleStateChanged(SessionState state)
        {
            switch (state)
            {
                case SessionState.Recording: ShowOnly(practicePanel);  break;
                case SessionState.Analyzing: ShowOnly(analyzingPanel); break;
                case SessionState.Complete:  ShowOnly(summaryPanel);   break;
                case SessionState.Error:     ShowOnly(summaryPanel);   break;
            }
        }

        private void ShowOnly(GameObject panel)
        {
            if (practicePanel != null)  practicePanel.SetActive(panel == practicePanel);
            if (analyzingPanel != null) analyzingPanel.SetActive(panel == analyzingPanel);
            if (summaryPanel != null)   summaryPanel.SetActive(panel == summaryPanel);
        }

        private void HandleError(string message)
        {
            if (feedbackText != null) feedbackText.text = $"เกิดข้อผิดพลาด: {message}";
        }

        // ============================================================== //
        //  Summary binding                                               //
        // ============================================================== //

        private void HandleSessionComplete(SessionData data, string feedback)
        {
            BindRadar(data);

            if (feedbackText != null)     feedbackText.text = feedback;
            if (overallScoreText != null) overallScoreText.text = $"{data.OverallScore():F0}";
            if (statsText != null)
                statsText.text =
                    $"{data.wordsPerMinute:F0} WPM   •   {data.fillerWordCount} fillers\n" +
                    $"{data.eyeContactPercentage:F0}% eye contact   •   {data.postureStatus}";
        }

        /// <summary>
        /// Pushes the five 0..100 scores into the XCharts RadarChart. We rebuild the
        /// indicators each time so axis labels/ranges always match SessionData.
        /// (XCharts 3.x API — adjust method names if you're on a different version.)
        /// </summary>
        private void BindRadar(SessionData data)
        {
            if (radarChart == null) return;

            // 1) Ensure exactly ONE radar serie (so repeated sessions don't stack series).
            if (radarChart.GetSerie(0) == null)
                radarChart.AddSerie<Radar>("You");

            // 2) ClearData() wipes serie data AND the radar indicators — RadarCoord.ClearData()
            //    calls indicatorList.Clear() — so indicators must be rebuilt AFTER this.
            radarChart.ClearData();

            // 3) Rebuild one spoke per axis, each on a fixed 0..100 range.
            var radarCoord = radarChart.EnsureChartComponent<RadarCoord>();
            radarCoord.indicatorList.Clear();
            foreach (var axis in SessionData.RadarAxes)
                radarCoord.AddIndicator(axis, 0, 100);

            // 4) Push the five scores as a single multidimensional data point.
            var scores = data.GetRadarScores();
            var values = new List<double>(scores.Count);
            foreach (var s in scores) values.Add(s);
            radarChart.AddData(0, values, "You");

            radarChart.RefreshChart();
        }
    }
}
