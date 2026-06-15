using System;
using UnityEngine;

// homuler MediaPipe Unity Plugin (Tasks API, v0.16.x).
using Mediapipe.Tasks.Components.Containers;
using Mediapipe.Tasks.Vision.FaceLandmarker;
using Mediapipe.Tasks.Vision.PoseLandmarker;

namespace PresentationCoach
{
    /// <summary>
    /// Owns everything camera-related. Consumes MediaPipe Tasks results and turns
    /// them into two coaching signals:
    ///   • Eye contact  — is the head oriented toward the camera? (from head pose)
    ///   • Posture      — Open vs Closed upper body (from pose landmarks)
    ///
    /// The heavy graph/inference setup lives in your MediaPipe runner; this class
    /// only needs the per-frame *results*. Hook the two Ingest* methods to your
    /// FaceLandmarker / PoseLandmarker result callbacks (see GRAPH WIRING below).
    /// </summary>
    public class VisionManager : MonoBehaviour
    {
        // ---- BlazePose 33-landmark indices we use ----
        private const int LEFT_SHOULDER = 11, RIGHT_SHOULDER = 12;

        [Header("Eye-contact thresholds (degrees)")]
        [Tooltip("Max head yaw (left/right) still counted as 'looking at camera'.")]
        [SerializeField] private float yawTolerance = 20f;
        [Tooltip("Max head pitch (up/down) still counted as 'looking at camera'.")]
        [SerializeField] private float pitchTolerance = 15f;

        [Header("Posture thresholds (world landmarks, meters)")]
        [Tooltip("Max shoulder height difference (m) before posture reads as slumped/leaning.")]
        [SerializeField] private float maxShoulderTiltMeters = 0.06f;
        [Tooltip("Max front-back gap between shoulders (m) before the torso reads as turned away.")]
        [SerializeField] private float maxShoulderDepthSkewMeters = 0.12f;
        [Tooltip("Min landmark visibility to trust a reading.")]
        [SerializeField] private float minVisibility = 0.5f;

        // ---- Live state ----
        public bool IsMakingEyeContact { get; private set; }
        public PostureStatus CurrentPosture { get; private set; } = PostureStatus.Unknown;
        public Vector3 LastHeadEuler { get; private set; }   // (pitch, yaw, roll) in degrees

        // ---- Session aggregates ----
        public float EyeContactPercentage  => _frames > 0 ? 100f * _eyeContactFrames / _frames : 0f;
        public float OpenPosturePercentage => _frames > 0 ? 100f * _openPostureFrames / _frames : 0f;

        // ---- Events ----
        public event Action<bool> OnEyeContactChanged;
        public event Action<PostureStatus> OnPostureChanged;

        private int _frames, _eyeContactFrames, _openPostureFrames;
        private bool _active;

        // ============================================================== //
        //  Session control                                               //
        // ============================================================== //

        public void BeginSession()
        {
            _frames = _eyeContactFrames = _openPostureFrames = 0;
            IsMakingEyeContact = false;
            CurrentPosture = PostureStatus.Unknown;
            _active = true;
        }

        public void EndSession() => _active = false;

        // ============================================================== //
        //  INTEGRATION POINTS — call from your MediaPipe result callbacks //
        // ============================================================== //
        //
        // GRAPH WIRING (pseudo-code, lives in your MediaPipe runner script):
        //
        //   var faceOpts = new FaceLandmarkerOptions(
        //       baseOptions: new BaseOptions(modelAssetPath: "face_landmarker.bytes"),
        //       runningMode: RunningMode.LIVE_STREAM,
        //       outputFaceTransformationMatrixes: true,        // <- needed for head pose
        //       resultCallback: (res, img, ts) => vision.IngestFaceResult(res));
        //   _faceLandmarker = FaceLandmarker.CreateFromOptions(faceOpts);
        //
        //   var poseOpts = new PoseLandmarkerOptions(
        //       baseOptions: new BaseOptions(modelAssetPath: "pose_landmarker.bytes"),
        //       runningMode: RunningMode.LIVE_STREAM,
        //       resultCallback: (res, img, ts) => vision.IngestPoseResult(res));
        //   _poseLandmarker = PoseLandmarker.CreateFromOptions(poseOpts);
        //
        //   // each camera frame:
        //   _faceLandmarker.DetectAsync(image, timestampMs);
        //   _poseLandmarker.DetectAsync(image, timestampMs);
        // ============================================================== //

        /// <summary>Hook to FaceLandmarker's result callback. Updates eye contact.</summary>
        public void IngestFaceResult(FaceLandmarkerResult result)
        {
            if (!_active) return;
            _frames++;

            // Prefer the rigid 4x4 facial transformation matrix: its rotation part
            // IS the head pose relative to the camera. (Requires
            // outputFaceTransformationMatrixes:true in the options.)
            if (result.facialTransformationMatrixes != null &&
                result.facialTransformationMatrixes.Count > 0)
            {
                Quaternion headRot = result.facialTransformationMatrixes[0].rotation;
                EvaluateEyeContact(headRot.eulerAngles);
            }
            else if (result.faceLandmarks != null && result.faceLandmarks.Count > 0)
            {
                // Fallback: estimate yaw from facial geometry when no matrix is output.
                EvaluateEyeContactFromLandmarks(result.faceLandmarks[0]);
            }
        }

        /// <summary>Hook to PoseLandmarker's result callback. Updates posture.</summary>
        public void IngestPoseResult(PoseLandmarkerResult result)
        {
            if (!_active) return;
            // Use WORLD landmarks (meters, origin at hip midpoint): aspect-ratio
            // invariant, unlike normalized image coords whose x/y use different scales.
            if (result.poseWorldLandmarks == null || result.poseWorldLandmarks.Count == 0) return;
            EvaluatePosture(result.poseWorldLandmarks[0]);
        }

        // ============================================================== //
        //  EYE CONTACT MATH                                              //
        // ============================================================== //
        //
        // The head pose gives Euler angles (pitch, yaw, roll). Eye contact is a
        // simple frustum test on yaw + pitch:
        //
        //     lookingAtCamera = |yaw| <= yawTolerance  AND  |pitch| <= pitchTolerance
        //
        // Yaw  = turning head left/right  (most important — "are they addressing me?")
        // Pitch= nodding up/down          (reading notes vs. facing audience)
        // Roll = head tilt                (ignored; doesn't break eye contact)
        //
        // eulerAngles come back in [0,360]; we wrap to [-180,180] so the threshold
        // is symmetric about straight-ahead (0°).
        // ============================================================== //

        private void EvaluateEyeContact(Vector3 euler)
        {
            float pitch = WrapDegrees(euler.x);
            float yaw   = WrapDegrees(euler.y);
            LastHeadEuler = new Vector3(pitch, yaw, WrapDegrees(euler.z));

            bool contact = Mathf.Abs(yaw) <= yawTolerance && Mathf.Abs(pitch) <= pitchTolerance;
            SetEyeContact(contact);
        }

        /// <summary>
        /// Geometry fallback: approximate yaw from the horizontal offset of the nose
        /// between the two ear/cheek landmarks. When facing forward the nose sits at
        /// the midpoint; turning the head pushes it toward one side.
        ///
        ///     offset = noseX - (leftX + rightX)/2          // signed, in [-w/2, w/2]
        ///     yaw   ≈ asin( 2·offset / faceWidth ) · Rad2Deg
        /// </summary>
        private void EvaluateEyeContactFromLandmarks(NormalizedLandmarks face)
        {
            var lm = face.landmarks;
            // Need index 454 valid (FaceMesh has 468/478 points); bail if model gave fewer.
            if (lm == null || lm.Count <= 454) { SetEyeContact(false); return; }

            // MediaPipe FaceMesh: 1 = nose tip, 234 = right cheek, 454 = left cheek.
            float noseX = lm[1].x;
            float rightX = lm[234].x, leftX = lm[454].x;
            float faceWidth = Mathf.Abs(leftX - rightX);
            if (faceWidth < 1e-4f) { SetEyeContact(false); return; }

            float offset = noseX - (leftX + rightX) * 0.5f;
            float yaw = Mathf.Asin(Mathf.Clamp(2f * offset / faceWidth, -1f, 1f)) * Mathf.Rad2Deg;
            LastHeadEuler = new Vector3(0f, yaw, 0f);
            SetEyeContact(Mathf.Abs(yaw) <= yawTolerance);
        }

        private void SetEyeContact(bool contact)
        {
            if (contact) _eyeContactFrames++;
            if (contact != IsMakingEyeContact)
            {
                IsMakingEyeContact = contact;
                OnEyeContactChanged?.Invoke(contact);
            }
        }

        // ============================================================== //
        //  POSTURE MATH  (world landmarks, in meters)                    //
        // ============================================================== //
        //
        // We classify Open vs Closed from the shoulder line using two cues that are
        // both scale- AND aspect-invariant (world landmarks are metric, so unlike
        // normalized image coords we don't have to worry about the camera's aspect
        // ratio distorting an x-vs-y ratio):
        //
        //   tilt      = |L_shoulder.y - R_shoulder.y|   -> leaning / slumping to a side
        //   depthSkew = |L_shoulder.z - R_shoulder.z|   -> torso rotated away from camera
        //
        //   OPEN   ⟺ tilt <= maxShoulderTiltMeters AND depthSkew <= maxShoulderDepthSkewMeters
        //   CLOSED otherwise (slumped, or turned so the shoulders aren't squared up)
        //
        // Rationale: a confident, engaged stance keeps the shoulders level and squared
        // to the audience. Leaning raises one shoulder (tilt); turning pushes one
        // shoulder closer to the camera than the other (depthSkew). Crossed-arm
        // detection is the natural extension — add wrist-vs-torso-midline checks.
        // ============================================================== //

        private void EvaluatePosture(Landmarks pose)
        {
            var lm = pose.landmarks;
            if (lm == null || lm.Count <= RIGHT_SHOULDER) return;
            if (!Visible(lm, LEFT_SHOULDER) || !Visible(lm, RIGHT_SHOULDER)) return;

            float tilt      = Mathf.Abs(lm[LEFT_SHOULDER].y - lm[RIGHT_SHOULDER].y);
            float depthSkew = Mathf.Abs(lm[LEFT_SHOULDER].z - lm[RIGHT_SHOULDER].z);

            bool open = tilt <= maxShoulderTiltMeters && depthSkew <= maxShoulderDepthSkewMeters;
            var status = open ? PostureStatus.Open : PostureStatus.Closed;

            if (open) _openPostureFrames++;
            if (status != CurrentPosture)
            {
                CurrentPosture = status;
                OnPostureChanged?.Invoke(status);
            }
        }

        // ---- small helpers ----

        private bool Visible(System.Collections.Generic.IReadOnlyList<Landmark> lm, int i)
            => !lm[i].visibility.HasValue || lm[i].visibility.Value >= minVisibility;

        /// <summary>Wraps an angle from [0,360] into [-180,180].</summary>
        private static float WrapDegrees(float deg)
        {
            deg %= 360f;
            if (deg > 180f) deg -= 360f;
            if (deg < -180f) deg += 360f;
            return deg;
        }
    }
}
