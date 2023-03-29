import './App.css';
import { OpenCvProvider, useOpenCv } from "opencv-react";
import React, { useState } from 'react';





function MyComponent() {
  const { loaded, cv } = useOpenCv();

  const [selectedImage, setSelectedImage] = useState(null);
  const [imageStatus, setImageStatus] = useState(null);
  const [src, setSrc] = useState(null);
  const [dst, setDst] = useState(null);

  const [grayScale, setGrayScale] = useState(true);
  const [edge, setEdge] = useState(true);
  const [rotate, setRotate] = useState(true);
  const [erosion, setErosion] = useState(true);
  const [dilation, setDialation] = useState(true);
  const [blurThresh, setBlurThresh] = useState(true);

  const onImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      let imgElement = document.getElementById("imageSrc");
      imgElement.src = URL.createObjectURL(e.target.files[0]);
      setImageStatus(imgElement);
    } else {
      console.log("File selection canceled.");
    }
  };

  const biggestContour = (contours, minArea) => {
    let maxArea = 0;
    let biggestN = 0;
    let approxContour = null;
    for (let n = 0; n < contours.size(); n++) {
      let i = contours.get(n);
      let area = cv.contourArea(i);
      if (area > minArea / 10) {
        let peri = cv.arcLength(i, true);
        let approx = new cv.Mat();
        cv.approxPolyDP(i, approx, 0.02 * peri, true);
        if (area > maxArea && approx.rows === 4) {
          maxArea = area;
          biggestN = n;
          approxContour = approx;
        }
        approx.delete();
      }
    }
    return { biggestN, approxContour };
  }

  const orderPoints = (pts) => {

    // Reshape pts to a 4x2 array
    pts = pts.reshape(4, 2);
    let rect = new cv.Mat.zeros(4, 2, cv.CV_32F);

    // Compute the sum of each row in pts
    let s = new cv.Mat();
    cv.reduce(pts, s, 1, cv.ReduceTypes.REDUCE_SUM);

    // Find the index of the point with the smallest and largest sum
    let minIdx = new cv.Mat();
    let maxIdx = new cv.Mat();
    cv.minMaxLoc(s, null, null, minIdx, maxIdx);

    // Set the top-left and bottom-right points of the rectangle
    rect.set(0, 0, pts.data32F[minIdx.data32S[0] * 2]);
    rect.set(0, 1, pts.data32F[minIdx.data32S[0] * 2 + 1]);
    rect.set(2, 0, pts.data32F[maxIdx.data32S[0] * 2]);
    rect.set(2, 1, pts.data32F[maxIdx.data32S[0] * 2 + 1]);

    // Compute the difference between each pair of points
    let diff = new cv.Mat();
    cv.absdiff(pts.row(0), pts.row(2), diff);

    // Find the index of the point with the smallest and largest difference
    cv.minMaxLoc(diff, null, null, minIdx, maxIdx);

    // Set the top-right and bottom-left points of the rectangle
    rect.set(1, 0, pts.data32F[minIdx.data32S[0] * 2]);
    rect.set(1, 1, pts.data32F[minIdx.data32S[0] * 2 + 1]);
    rect.set(3, 0, pts.data32F[maxIdx.data32S[0] * 2]);
    rect.set(3, 1, pts.data32F[maxIdx.data32S[0] * 2 + 1]);

    // Free up memory
    s.delete();
    minIdx.delete();
    maxIdx.delete();
    diff.delete();

    return rect;
  }

  const fourPointTransform = (image, pts) => {
    // obtain a consistent order of the points and unpack them
    // individually
    const rect = orderPoints(pts);
    const [tl, tr, br, bl] = rect;

    // compute the width of the new image, which will be the
    // maximum distance between bottom-right and bottom-left
    // x-coordinates or the top-right and top-left x-coordinates
    const widthA = Math.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2));
    const widthB = Math.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2));
    const maxWidth = Math.max(Math.round(widthA), Math.round(widthB));


    // compute the height of the new image, which will be the
    // maximum distance between the top-right and bottom-right
    // y-coordinates or the top-left and bottom-left y-coordinates
    const heightA = Math.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2));
    const heightB = Math.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2));
    const maxHeight = Math.max(Math.round(heightA), Math.round(heightB));

    // now that we have the dimensions of the new image, construct
    // the set of destination points to obtain a "birds eye view",
    // (i.e. top-down view) of the image, again specifying points
    // in the top-left, top-right, bottom-right, and bottom-left
    // order
    const dst = [
      [0, 0],
      [maxWidth - 1, 0],
      [maxWidth - 1, maxHeight - 1],
      [0, maxHeight - 1]
    ];

    // compute the perspective transform matrix and then apply it
    const M = cv.getPerspectiveTransform(rect, dst);
    const warped = cv.warpPerspective(image, M, [maxWidth, maxHeight]);

    // return the warped image
    return warped;
  }

  // const increaseBrightness = (img, value = 30) => {
  //   const hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV);
  //   const [h, s, v] = cv.split(hsv);
  //   const lim = 255 - value;
  //   v[v > lim] = 255;
  //   v[v <= lim] += value;
  //   const final_hsv = cv.merge([h, s, v]);
  //   const resultImg = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR);
  //   return resultImg;
  // }

  // const finalImage = (rotated) => {
  //   const kernelSharpening = [
  //     [0, -1, 0],
  //     [-1, 5, -1],
  //     [0, -1, 0]];
  //   const sharpened = cv.filter2D(rotated, -1, kernelSharpening.tolist())
  //   const finalImage = increaseBrightness(sharpened, 30)
  //   return finalImage
  // }

  const onBlurThreshold = () => {
    setBlurThresh(!blurThresh);
    if (blurThresh) {
      let s = cv.imread(imageStatus);
      let d1 = new cv.Mat();
      let d2 = new cv.Mat();
      let d3 = new cv.Mat();

      setSrc(s);
      setDst(d1);

      cv.cvtColor(s, d1, cv.COLOR_RGBA2GRAY, 0);
      cv.bilateralFilter(d1, d2, 11, 31, 9);
      cv.adaptiveThreshold(d2, d3, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2);

      //a partir de aca es todo recortar
      const image_size = d3.size;
      // const edges = cv.Canny(d3, 50, 150, { apertureSize: 7 });
      // const contours = [];
      // const hierarchy = [];
      // cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
      // const simplified_contours = [];
      // contours.forEach(cnt => {
      //   const hull = cv.convexHull(cnt)
      //   simplified_contours.push(cv.approxPolyDP(hull, 0.001 * cv.arcLength(hull, true), true))
      // });

      // const simplified_contours_np = new Array(simplified_contours);
      // const [biggest_n, approx_contour] = biggestContour(simplified_contours_np, image_size)

      // const thresholdedImage = cv.drawContours(d3, simplified_contours, biggest_n, [0, 255, 0], 1)
      // let a = 0
      // if (approx_contour !== null && approx_contour.length === 4) {
      //   a = fourPointTransform(thresholdedImage, approx_contour)
      // }

      // const croppedImage = a
      cv.imshow("canvasOutput", d3);
      // console.log(simplified_contours);
      //setImageStatus(dst);
      //src.delete();
      //dst.delete();
    }
  };







  const onGrayScaleChange = () => {
    setGrayScale(!grayScale);
    if (grayScale) {
      let s = cv.imread(imageStatus);
      let d = new cv.Mat();
      setSrc(s);
      setDst(d);

      // You can try more different parameters
      // console.log(src);
      // console.log(dst);
      cv.cvtColor(s, d, cv.COLOR_RGBA2GRAY, 0);
      cv.imshow("canvasOutput", d);
      //setImageStatus(dst);
      //src.delete();
      //dst.delete();
    }
  };

  const onEdgeChange = () => {
    setEdge(!edge);
    if (edge) {
      let s = cv.imread(imageStatus);
      let d = new cv.Mat();
      setSrc(s);
      setDst(d);

      cv.cvtColor(src, src, cv.COLOR_RGB2GRAY, 0);
      // You can try more different parameters
      cv.Canny(src, dst, 50, 100, 3, false);
      cv.imshow("canvasOutput", dst);
      // src.delete();
      // dst.delete();
    }
  };

  const onRotateChange = () => {
    setRotate(!rotate);
    if (rotate) {
      let s = cv.imread(imageStatus);
      let d = new cv.Mat();
      setSrc(s);
      setDst(d);

      let dsize = new cv.Size(src.rows, src.cols);
      let center = new cv.Point(src.cols / 2, src.rows / 2);
      // You can try more different parameters
      let M = cv.getRotationMatrix2D(center, 90, 1);
      cv.warpAffine(
        src,
        dst,
        M,
        dsize,
        cv.INTER_LINEAR,
        cv.BORDER_CONSTANT,
        new cv.Scalar()
      );
      cv.imshow("canvasOutput", dst);
      //src.delete();
      //dst.delete();
      M.delete();
    }
  };

  const onErosionChange = () => {
    setErosion(!erosion);
    if (erosion) {
      let s = cv.imread(imageStatus);
      let d = new cv.Mat();
      setSrc(s);
      setDst(d);

      let M = cv.Mat.ones(5, 5, cv.CV_8U);
      let anchor = new cv.Point(-1, -1);
      // You can try more different parameters
      cv.erode(
        src,
        dst,
        M,
        anchor,
        1,
        cv.BORDER_CONSTANT,
        cv.morphologyDefaultBorderValue()
      );
      cv.imshow("canvasOutput", dst);
    }
  };

  const onDilationChange = () => {
    setDialation(!dilation);
    if (dilation) {
      let s = cv.imread(imageStatus);
      let d = new cv.Mat();
      setSrc(s);
      setDst(d);

      let M = cv.Mat.ones(5, 5, cv.CV_8U);
      let anchor = new cv.Point(-1, -1);
      // You can try more different parameters
      cv.dilate(
        src,
        dst,
        M,
        anchor,
        1,
        cv.BORDER_CONSTANT,
        cv.morphologyDefaultBorderValue()
      );
      cv.imshow("canvasOutput", dst);
    }
  };







  if (loaded) {
    return (
      <>
        <div className="inputoutput">
          <div className="processing">
            Apply Grayscale
            <input
              type="checkbox"
              id="RGB2Gray"
              name="RGB2Gray"
              value="Grayscale Conversion"
              onChange={onGrayScaleChange}
            />
            <br></br>
            Detect Edges
            <input
              type="checkbox"
              id="edgeDetection"
              name="edgeDetection"
              value="Edge Detection"
              onChange={onEdgeChange}
            />
            <br></br>
            Rotate Image
            <input
              type="checkbox"
              id="rotateImage"
              name="rotateImage"
              value="Rotate Image"
              onChange={onRotateChange}
            />
            <br></br>
            Image Erosion
            <input
              type="checkbox"
              id="erosion"
              name="erosion"
              value="Image Erosion"
              onChange={onErosionChange}
            />
            <br></br>
            Image Dilation
            <input
              type="checkbox"
              id="dilation"
              name="dilation"
              value="Image Dilation"
              onChange={onDilationChange}
            />
            <br></br>
            Blur and Threshold
            <input
              type="checkbox"
              id="threshold"
              name="threshold"
              value="Blur and Threshold"
              onChange={onBlurThreshold}
            />
          </div>
          <img id="imageSrc" alt="Nada" />
          <div className="caption">
            imageSrc{" "}
            <input
              type="file"
              id="fileInput"
              name="file"
              onChange={(e) => onImageChange(e)}
            />
          </div>
        </div>
        <div className="inputoutput">
          <canvas id="canvasOutput"></canvas>
          <div className="caption">canvasOutput</div>
        </div>
      </>
    );
    // return <p>opencv loaded</p>;
  } else {
    return (
      <div>
        <h1>Testing Opencv-React Lib </h1>
        {selectedImage && (
          <div>
            <img
              alt="not fount"
              width={"250px"}
              src={URL.createObjectURL(selectedImage)}
              id="img"
            />
            <br />
            <button onClick={() => setSelectedImage(null)}>Remove</button>
            <canvas id="output"></canvas>
          </div>
        )}
        <br />

        <br />
        <input
          type="file"
          name="myImage"
          onChange={(event) => {
            console.log(event.target.files[0]);
            setSelectedImage(event.target.files[0]);
          }}
        />
      </div>
    );
  }
}

const App = () => {
  const onLoaded = (cv) => {
    console.log("opencv loaded, cv");
  };

  return (
    <OpenCvProvider onLoad={onLoaded} openCvPath="/opencv/opencv.js">
      <MyComponent />
    </OpenCvProvider>
  );
};


export default App;
