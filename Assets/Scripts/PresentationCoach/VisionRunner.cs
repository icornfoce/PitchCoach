using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

// homuler MediaPipe Unity Plugin (Tasks API, v0.16.x).
using MpImage = Mediapipe.Image;                  // alias: avoids clash with UnityEngine.UI.Image
using Mediapipe.Unity.Experimental;               // TextureFramePool, TextureFrame
using Mediapipe.Tasks.Core;                       // BaseOptions
using Mediapipe.Tasks.Vision.Core;                // RunningMode, ImageProcessingOptions
using Mediapipe.Tasks.Vision.FaceLandmarker;
using Mediapipe.Tasks.Vision.PoseLandmarker;
using Stopwatch = System.Diagnostics.Stopwatch;

namespace PresentationCoach
{
    /// <summary>
    /// The MediaPipe "runner" that <see cref="VisionManager"/> depends on. It owns the
    /// camera and the two inference graphs, and feeds their results back to VisionManager:
    ///
    ///   webcam frame ─► TextureFramePool ─► Image ─► FaceLandmarker.DetectAsync ─┐
    ///                                              └► PoseLandmarker.DetectAsync ─┤
    ///                                                                            ▼
    ///                          (worker-thread result callbacks, buffered)        │
    ///   VisionManager.IngestFaceResult / IngestPoseResult  ◄── Update() flush ───┘
    ///
    /// MediaPipe LIVE_STREAM callbacks arrive on worker threads, so results are stashed
    /// under a lock and replayed to VisionManager in <see cref="Update"/> (main thread),
    /// where the downstream events safely touch Unity UI. This mirrors the pattern in
    /// homuler's official LIVE_STREAM samples (callback stores → main thread draws later).
    ///
    /// Models (.bytes) are read straight from StreamingAssets and passed as a byte buffer,
    /// so no MediaPipe sample bootstrap / ResourceManager is required.
    /// </summary>
    public class VisionRunner : MonoBehaviour
    {
        [Header("Wiring")]
        [SerializeField] private VisionManager visionManager;
        [Tooltip("Optional: displays the live camera feed in the practice panel.")]
        [SerializeField] private RawImage cameraPreview;

        [Header("Camera")]
        [SerializeField] private int requestedWidth = 1280;
        [SerializeField] private int requestedHeight = 720;
        [SerializeField] private int requestedFps = 30;
        [Tooltip("Index into WebCamTexture.devices.")]
        [SerializeField] private int cameraDeviceIndex = 0;

        [Header("Image orientation")]
        [Tooltip("Metrics are flip-invariant; these mainly affect detection quality + the preview.")]
        [SerializeField] private bool flipHorizontally = false;
        [SerializeField] private bool flipVertically = true;

        [Header("Models (file names under StreamingAssets)")]
        [SerializeField] private string faceModelFile = "face_landmarker_v2.bytes";
        [SerializeField] private string poseModelFile = "pose_landmarker_full.bytes";

        [Header("Detection confidence")]
        [Range(0f, 1f)] [SerializeField] private float minDetectionConfidence = 0.5f;
        [Range(0f, 1f)] [SerializeField] private float minPresenceConfidence = 0.5f;
        [Range(0f, 1f)] [SerializeField] private float minTrackingConfidence = 0.5f;

        // ---- MediaPipe task handles ----
        private FaceLandmarker _faceLandmarker;
        private PoseLandmarker _poseLandmarker;

        // ---- Camera ----
        private WebCamTexture _webCamTexture;
        private TextureFramePool _framePool;

        // ---- Monotonic millisecond timestamp source for LIVE_STREAM ----
        private readonly Stopwatch _stopwatch = new Stopwatch();
        private long _lastTimestamp = -1;

        // ---- Worker-thread -> main-thread handoff ----
        private readonly object _lock = new object();
        private FaceLandmarkerResult _pendingFace; private bool _hasFace;
        private PoseLandmarkerResult _pendingPose; private bool _hasPose;

        private bool _running;

        // ============================================================== //
        //  Setup                                                         //
        // ============================================================== //

        private IEnumerator Start()
        {
            if (visionManager == null)
            {
                Debug.LogError("[VisionRunner] VisionManager reference is not assigned.");
                yield break;
            }

            if (!TryCreateLandmarkers())
                yield break;   // error already logged

            if (WebCamTexture.devices.Length == 0)
            {
                Debug.LogError("[VisionRunner] No webcam devices found.");
                yield break;
            }

            int idx = Mathf.Clamp(cameraDeviceIndex, 0, WebCamTexture.devices.Length - 1);
            var device = WebCamTexture.devices[idx];
            _webCamTexture = new WebCamTexture(device.name, requestedWidth, requestedHeight, requestedFps);
            _webCamTexture.Play();

            // The texture has a dummy 16x16 size until the first real frame arrives.
            yield return new WaitUntil(() => _webCamTexture.width > 16);

            if (cameraPreview != null) cameraPreview.texture = _webCamTexture;

            _framePool = new TextureFramePool(_webCamTexture.width, _webCamTexture.height, TextureFormat.RGBA32, 10);
            _stopwatch.Start();
            _running = true;

            var imageProcessingOptions = new ImageProcessingOptions(rotationDegrees: 0);

            // Alternate FACE then POSE every iteration; both run off the same live feed.
            while (_running)
            {
                yield return CaptureAndDetect(true, imageProcessingOptions);   // face
                yield return CaptureAndDetect(false, imageProcessingOptions);  // pose
            }
        }

        private bool TryCreateLandmarkers()
        {
            try
            {
                byte[] faceBytes = LoadModel(faceModelFile);
                byte[] poseBytes = LoadModel(poseModelFile);

                // CPU delegate: GPU inference isn't supported in the Windows/macOS editor.
                var faceOptions = new FaceLandmarkerOptions(
                    new BaseOptions(BaseOptions.Delegate.CPU, modelAssetBuffer: faceBytes),
                    runningMode: RunningMode.LIVE_STREAM,
                    numFaces: 1,
                    minFaceDetectionConfidence: minDetectionConfidence,
                    minFacePresenceConfidence: minPresenceConfidence,
                    minTrackingConfidence: minTrackingConfidence,
                    outputFaceBlendshapes: false,
                    outputFaceTransformationMatrixes: true,   // <- head pose for eye contact
                    resultCallback: OnFaceResult);
                _faceLandmarker = FaceLandmarker.CreateFromOptions(faceOptions);

                var poseOptions = new PoseLandmarkerOptions(
                    new BaseOptions(BaseOptions.Delegate.CPU, modelAssetBuffer: poseBytes),
                    runningMode: RunningMode.LIVE_STREAM,
                    numPoses: 1,
                    minPoseDetectionConfidence: minDetectionConfidence,
                    minPosePresenceConfidence: minPresenceConfidence,
                    minTrackingConfidence: minTrackingConfidence,
                    outputSegmentationMasks: false,
                    resultCallback: OnPoseResult);
                _poseLandmarker = PoseLandmarker.CreateFromOptions(poseOptions);

                return true;
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[VisionRunner] Failed to create MediaPipe landmarkers: {e.Message}\n{e}");
                return false;
            }
        }

        private static byte[] LoadModel(string fileName)
        {
            string path = Path.Combine(Application.streamingAssetsPath, fileName);
            if (!File.Exists(path))
                throw new FileNotFoundException(
                    $"[VisionRunner] Model '{fileName}' not found under StreamingAssets. " +
                    $"Copy it from the MediaPipe package's PackageResources/MediaPipe folder.", path);
            return File.ReadAllBytes(path);
        }

        // ============================================================== //
        //  Per-frame capture + inference                                 //
        // ============================================================== //

        private IEnumerator CaptureAndDetect(bool face, ImageProcessingOptions ipo)
        {
            if (_framePool == null || !_framePool.TryGetTextureFrame(out var frame))
            {
                yield return null;   // pool exhausted this frame; try again next frame
                yield break;
            }

            // Async GPU readback of the webcam into a TextureFrame (handles flips correctly).
            var req = frame.ReadTextureAsync(_webCamTexture, flipHorizontally, flipVertically);
            yield return new WaitUntil(() => req.done);

            if (req.hasError)
            {
                frame.Release();
                yield break;
            }

            // BuildCPUImage snapshots the pixels into an Image; the frame can be recycled.
            MpImage image = frame.BuildCPUImage();
            frame.Release();

            long ts = NextTimestamp();
            if (face) _faceLandmarker.DetectAsync(image, ts, ipo);
            else      _poseLandmarker.DetectAsync(image, ts, ipo);
        }

        /// <summary>Strictly-increasing millisecond timestamp (LIVE_STREAM requires monotonicity).</summary>
        private long NextTimestamp()
        {
            long ts = _stopwatch.ElapsedTicks / System.TimeSpan.TicksPerMillisecond;
            if (ts <= _lastTimestamp) ts = _lastTimestamp + 1;
            _lastTimestamp = ts;
            return ts;
        }

        // ============================================================== //
        //  Result callbacks (worker thread) -> buffer                    //
        // ============================================================== //

        private void OnFaceResult(FaceLandmarkerResult result, MpImage image, long timestamp)
        {
            lock (_lock) { _pendingFace = result; _hasFace = true; }
        }

        private void OnPoseResult(PoseLandmarkerResult result, MpImage image, long timestamp)
        {
            lock (_lock) { _pendingPose = result; _hasPose = true; }
        }

        // ============================================================== //
        //  Main-thread flush -> VisionManager                            //
        // ============================================================== //

        private void Update()
        {
            if (!_running || visionManager == null) return;

            FaceLandmarkerResult face = default; bool hasFace = false;
            PoseLandmarkerResult pose = default; bool hasPose = false;

            lock (_lock)
            {
                if (_hasFace) { face = _pendingFace; hasFace = true; _hasFace = false; }
                if (_hasPose) { pose = _pendingPose; hasPose = true; _hasPose = false; }
            }

            if (hasFace) visionManager.IngestFaceResult(face);
            if (hasPose) visionManager.IngestPoseResult(pose);
        }

        // ============================================================== //
        //  Teardown                                                      //
        // ============================================================== //

        private void OnDestroy()
        {
            _running = false;

            if (_webCamTexture != null)
            {
                if (_webCamTexture.isPlaying) _webCamTexture.Stop();
                _webCamTexture = null;
            }

            _faceLandmarker?.Close(); _faceLandmarker = null;
            _poseLandmarker?.Close(); _poseLandmarker = null;
            _framePool?.Dispose(); _framePool = null;

            if (_stopwatch.IsRunning) _stopwatch.Stop();
        }
    }
}
