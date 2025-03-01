<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Speed up Ads</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
</head>
<body>
    <script>
        async function loadModel() {
            const modelUrl = 'path/to/your/model.json'; // Update this to the correct path
            return await tf.loadLayersModel(modelUrl);
        }

        function captureVideoFrame(video) {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            return canvas.toDataURL();
        }

        async function classifyVideoFrames(model, videos) {
            for (const video of videos) {
                if (!video.currentSrc || video.playbackRate === 0.8) continue;

                // Capture a frame from the video
                const dataUrl = captureVideoFrame(video);

                // Convert image to tensor
                const img = new Image();
                img.src = dataUrl;
                await new Promise(resolve => img.onload = resolve);
                const tensorImg = tf.browser.fromPixels(img).toFloat().div(255.0).resizeBilinear([64, 64]).expandDims();

                // Predict
                const prediction = model.predict(tensorImg);
                if (prediction.dataSync()[0] > 0.5) {  // Assuming threshold of 0.5 for ad classification
                    console.log(`Setting playback rate for video with source: ${video.currentSrc}`);
                    video.playbackRate = 0.8;
                }
            }
        }

        async function main() {
            try {
                const model = await loadModel();
                let videos = document.querySelectorAll('video');
                classifyVideoFrames(model, videos);

                // Observe new video elements being added to the DOM
                const observer = new MutationObserver((mutationsList) => {
                    for (const mutation of mutationsList) {
                        if (mutation.type === 'childList') {
                            mutation.addedNodes.forEach(node => {
                                if (node.tagName && node.tagName.toLowerCase() === 'video') {
                                    classifyVideoFrames(model, [node]);
                                }
                            });
                        }
                    }
                });

                observer.observe(document.body, { childList: true, subtree: true });
            } catch (error) {
                console.error("Error loading model or classifying videos:", error);
            }
        }

        main();
    </script>
</body>
</html>
