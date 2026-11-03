/* Camera bridge for Ka-myii's 3D workspace. The face tracker and all model files stay local. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(root);
  else root.Kamyii3DTracking = factory(root);
})(typeof window !== 'undefined' ? window : globalThis, function (root) {
  'use strict';
  const BASE = '/static/vendor/anime25d/lib/vendor/face_mesh/';
  let meshPromise = null;

  // RigDevices normally resolves FaceMesh relative to the 2D preview's <base>. Give it the
  // same bundled files through an absolute path here without changing the 3D page's base URL.
  function ensureFaceMesh() {
    if (meshPromise) return meshPromise;
    meshPromise = new Promise((resolve, reject) => {
      function wrap() {
        const FaceMesh = root.FaceMesh;
        if (!FaceMesh) { reject(new Error('The offline face tracker could not load.')); return; }
        if (!FaceMesh._kamyiiAbsoluteAssets) {
          const LocalFaceMesh = function (options) {
            return new FaceMesh(Object.assign({}, options, { locateFile: name => {
              if (/wasm_bin/.test(name) && !/simd/.test(name))
                throw new Error('Offline face tracking needs WebAssembly SIMD.');
              return BASE + name;
            }}));
          };
          LocalFaceMesh._kamyiiAbsoluteAssets = true;
          root.FaceMesh = LocalFaceMesh;
        }
        resolve();
      }
      if (root.FaceMesh) { wrap(); return; }
      const script = document.createElement('script');
      script.src = BASE + 'face_mesh.js'; script.async = true;
      const timer = setTimeout(() => { script.remove(); reject(new Error('Face tracking model loading timed out.')); }, 8000);
      script.onload = () => { clearTimeout(timer); wrap(); };
      script.onerror = () => { clearTimeout(timer); script.remove(); reject(new Error('The offline face tracking model could not load.')); };
      document.head.appendChild(script);
    }).catch(err => { meshPromise = null; throw err; });
    return meshPromise;
  }

  function create(opts) {
    opts = opts || {};
    const devices = opts.devices || root.RigDevices;
    const features = opts.features || root.FaceFeatures;
    if (!devices || !features) throw new Error('The local face-tracking scripts are unavailable.');
    const tracker = new features.Tracker({linkEyes:false});
    const now = opts.now || (() => performance.now());
    const state = (s,m) => { if (opts.onState) opts.onState(s,m || ''); };
    const pose = value => { if (opts.onPose) opts.onPose(value); };
    let generation = 0, lastSeen = 0, stale = true, ticker = null, calibration = null;
    const camera = devices.createCamera({
      onState(s,msg) {
        if (s === 'on') { tracker.reset(); stale = true; lastSeen = 0; }
        if (s === 'error') {
          if (ticker) clearInterval(ticker); ticker = null;
          stopCalibration(); lastSeen = 0; stale = true; pose(null);
        }
        state(s,msg);
      },
      onResults(lm, video) {
        const aspect = video.videoWidth && video.videoHeight ? video.videoWidth / video.videoHeight : 4/3;
        const result = lm ? tracker.update(lm, now()/1000, aspect) : null;
        if (!result) return;
        lastSeen = now();
        if (stale) { stale = false; state('tracking','Face detected'); }
        pose(result);
      }
    });
    function stopCalibration() {
      if (!calibration) return;
      clearInterval(calibration.timer); tracker.samples = null;
      calibration.resolve(false); calibration = null;
    }
    function stop() {
      generation++;
      if (ticker) clearInterval(ticker); ticker = null;
      stopCalibration();
      camera.stop(); tracker.reset(); lastSeen = 0; stale = true; pose(null);
    }
    async function start() {
      // Starting twice releases the previous stream and invalidates pending permission requests.
      stop();
      const token = ++generation;
      state('loading','Starting local face tracking…');
      try {
        await (opts.ensureFaceMesh || ensureFaceMesh)();
        if (token !== generation) return false;
        const started = await camera.start();
        if (token !== generation || !started) return false;
        const startedAt = now();
        ticker = setInterval(() => {
          if (token !== generation) return;
          if (!stale && now() - lastSeen >= 1200) {
            stale = true; pose(null); state('no-face','Face not detected');
          } else if (stale && !lastSeen && now() - startedAt >= 1200) {
            lastSeen = -1; state('no-face','Face not detected');
          }
        }, 200);
        return true;
      } catch (error) {
        if (token !== generation) return false;
        camera.stop(true); state('error', error && error.message || 'Unable to start the camera.');
        throw error;
      }
    }
    function calibrate() {
      if (!camera.active) return Promise.resolve(false);
      stopCalibration(); tracker.startCalibration();
      state('calibrating','Look straight at the camera with a relaxed expression.');
      const startedAt = now();
      return new Promise(resolve => {
        const timer = setInterval(() => {
          const elapsed = now() - startedAt, count = tracker.samples ? tracker.samples.length : 0;
          if (camera.active && elapsed < 6000 && (elapsed < 1000 || count < 5)) return;
          clearInterval(timer); calibration = null;
          if (!camera.active) { tracker.samples = null; resolve(false); return; }
          const result = tracker.finishCalibration();
          tracker.reset();
          state(result ? 'tracking' : 'no-face', result ? 'Neutral pose calibrated' : 'No face detected for calibration');
          resolve(!!result);
        }, 100);
        calibration = {timer,resolve};
      });
    }
    return {start,stop,calibrate,dispose:stop,get active(){return camera.active;}};
  }
  return {create};
});
