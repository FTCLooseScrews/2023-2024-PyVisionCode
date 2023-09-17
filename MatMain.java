import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.markers.SeriesMarkers;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.awt.*;
import java.io.IOException;
import java.util.*;
import java.util.List;

public class MatMain {

    // reading image
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat image = Imgcodecs.imread("pic1.jpg");
        Imgproc.resize(image, image, new Size(640, 480));

        //blur image with gaussian blurring
        Mat blurredImage = new Mat();
        Imgproc.GaussianBlur(image, blurredImage, new Size(15, 15),0, 0);

        Imgcodecs.imwrite("output.jpg", applyKmeans(blurredImage, findOptimalK(blurredImage)));
    }

    public static int findOptimalK(Mat image) {
        int maxK = 21; // Maximum value of K to consider
        int bestK = 1; // Initialize with K=1
        double[] wcssValues = new double[maxK];

        image.convertTo(image, CvType.CV_32F);
        Mat data = image.reshape(1, (int) image.total());

        for (int k = 2; k <= maxK; k++) {
            Mat bestLabels = new Mat();
            TermCriteria criteria = new TermCriteria();
            int attempts = 5;
            int flags = Core.KMEANS_RANDOM_CENTERS;
            Mat centers = new Mat();

            Core.kmeans(data, k, bestLabels, criteria, attempts, flags, centers);

            // Calculate WCSS for this K
            double wcss = 0;
            for (int i = 0; i < data.rows(); i++) {
                int label = (int) bestLabels.get(i, 0)[0];
                double[] center = centers.get(label, 0);
                double[] point = data.get(i, 0);
                wcss += Math.pow(point[0] - center[0], 2);
            }

            wcssValues[k - 1] = wcss;
        }

        double minAngle = 0;


        // for each value of K starting at 2, find the K value that has an angle closest to 90 degrees.
        for (int k = 2; k < maxK-1; k++) {
            if (wcssValues[k] > wcssValues[k - 1]) continue;
            double angle = findAngle(1, (wcssValues[k] - wcssValues[k-1])/1E8, 1, (wcssValues[k+1] - wcssValues[k])/1E8);
            if (Math.abs(Math.PI/2 - angle) < Math.abs(Math.PI / 2 - minAngle)) {
                minAngle = angle;
                bestK = k+1;
            }
        }

        saveElbowGraph(wcssValues);

        System.out.println("Optimal K: " + bestK);

        return bestK;
    }

    // method for finding angle in the elbow graph
    public static double findAngle(double x1, double y1, double x2, double y2) {
        double dotProduct = x1 * x2 + y1 * y2;
        double magAB = Math.sqrt(Math.pow(x1, 2) + Math.pow(y1, 2)) * Math.sqrt(Math.pow(x2, 2) + Math.pow(y2, 2));
        return Math.acos(dotProduct/magAB);
    }

    // saving elbow graph as a seperate picture
    public static void saveElbowGraph(double[] wcssValues) {
        int maxClusters = wcssValues.length;
        int[] x = new int[maxClusters];
        double[] y = new double[maxClusters];

        for (int i = 0; i < maxClusters; i++) {
            x[i] = i + 1;
            y[i] = wcssValues[i];
        }

        // Create an XYChart (line chart) for the elbow graph
        XYChart chart = new XYChartBuilder()
                .width(800)
                .height(600)
                .title("Elbow Method For Optimal k")
                .xAxisTitle("k")
                .yAxisTitle("Sum_of_squared_distances")
                .build();

        double[] doubleX = new double[maxClusters];
        for (int i = 0; i < x.length; i++) {
            doubleX[i] = x[i];
        }

        chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line);
        chart.addSeries("Elbow", doubleX, y).setMarker(SeriesMarkers.NONE);

        try {
            // Save the chart as a PNG image
            BitmapEncoder.saveBitmap(chart, "ElbowGraphOpenCV.png", BitmapEncoder.BitmapFormat.PNG);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // using the optimal K, apply kmeans clustering onto the image
    public static Mat applyKmeans(Mat image, int optimalK) {
        image.convertTo(image, CvType.CV_32F);
        Mat data = image.reshape(1, (int) image.total());

        Mat bestLabels = new Mat();
        TermCriteria criteria = new TermCriteria();
        int attempts = 5;
        int flags = Core.KMEANS_RANDOM_CENTERS;
        Mat centers = new Mat();

        // perform kmeans
        Core.kmeans(data, optimalK, bestLabels, criteria, attempts, flags, centers);

        Mat draw = new Mat((int) image.total(), 1, CvType.CV_32FC3);
        Mat colors = centers.reshape(3, optimalK);
        for (int i = 0; i < optimalK; i++) {
            Mat mask = new Mat(); // a mask for each cluster label
            Core.compare(bestLabels, new Scalar(i), mask, Core.CMP_EQ);
            Mat col = colors.row(i); // can't use the Mat directly with setTo() (see #19100)
            double d[] = col.get(0, 0); // can't create Scalar directly from get(), 3 vs 4 elements
            draw.setTo(new Scalar(d[0], d[1], d[2]), mask);
        }

        //creates picture after K-Means has separated it into an image containing the optimal values of colors
        draw = draw.reshape(3, image.rows());
        draw.convertTo(draw, CvType.CV_8U);

        //get the optimal values of colors
        List<Color> dominantColors = getDominantColors(draw);

        //find which color is the 'red' color
        Color closestToRed = findClosestColor(new Color(255, 0, 0), dominantColors);

        System.out.printf("Closest Value To Red: RGB(%d, %d, %d)%n", closestToRed.getRed(), closestToRed.getGreen(), closestToRed.getBlue());

        //Now we are thresholding the image on our red color
        Scalar upper = new Scalar(closestToRed.getRed(), closestToRed.getGreen(), closestToRed.getBlue());

        Mat thresh = new Mat();
        Mat src = new Mat();
        Imgproc.cvtColor(draw, src, Imgproc.COLOR_BGR2RGB);
        Core.inRange(src, upper, upper, thresh);

        //Blur our threshold on a 13x13 kernel
        Mat blurredImage = new Mat();
        Imgproc.GaussianBlur(thresh, blurredImage, new Size(13, 13),0, 0);

        //Dilate our threshold on a 7x7 kernel
        Mat dilatedImage = new Mat();
        Mat kernel = Imgproc.getStructuringElement(Imgproc.CV_SHAPE_RECT, new Size(7, 7));
        Imgproc.dilate(blurredImage, dilatedImage, kernel, new Point(-1, -1), 4);

        //Find contours around our red threshold
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(dilatedImage, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        Mat img = new Mat();
        image.copyTo(img);

        //Draw our contours on our original blurred image (the one with colors)
        Imgproc.drawContours(img, contours, -1, new Scalar(0, 255, 0), 5);

        //Find out which contour has the maximum area (the one with maximum area will be our team prop)
        LinkedList<Double> areaList = new LinkedList<>();
        for (MatOfPoint contour : contours) areaList.add(Imgproc.contourArea(contour));

        double maxArea = 0;
        for (double area : areaList) maxArea = Math.max(area, maxArea);

        MatOfPoint propContour = contours.get(areaList.indexOf(maxArea));
        Imgproc.drawContours(img, Collections.singletonList(propContour), -1, new Scalar(255, 0, 255), 5);

        //Compare contour x coordinate with the other two contours to figure out which position team prop is in
        LinkedList<Double> xPosList = new LinkedList<>();

        for (MatOfPoint contour : contours) xPosList.add(contour.toList().get(0).x);
        Collections.sort(xPosList);

        int index = xPosList.indexOf(propContour.toList().get(0).x);

        //Print out position
        System.out.println("Position is: " + (index == 0 ? "Left" : index == 1 ? "Middle" : "Right"));

        Imgcodecs.imwrite("contours.jpg", img);
        return draw;
    }


    // get the colors from the kmeans clustered image
    private static List<Color> getDominantColors(Mat img) {
        HashSet<String> uniqueColors = new HashSet<>();
        HashSet<Scalar> colors = new HashSet<>();

        // Loop through each pixel in the image
        for (int i = 0; i < img.rows(); i++) {
            for (int j = 0; j < img.cols(); j++) {
                // Get the BGR color value at this pixel
                double[] pixel = img.get(i, j);

                // Convert BGR to RGB
                int blue = (int) pixel[0];
                int green = (int) pixel[1];
                int red = (int) pixel[2];

                // Create an RGB color string
                String color = String.format("RGB(%d, %d, %d)", red, green, blue);

                // Add the color to the HashSet
                colors.add(new Scalar(red, green, blue));
                uniqueColors.add(color);
            }
        }

        //Print out rgb values
        Iterator<String> it = uniqueColors.iterator();
        while (it.hasNext()) {
            System.out.println(it.next() + " ");
        }

        LinkedList<Color> ret = new LinkedList<>();

        //Add all dominant colors to return value list
        Iterator<Scalar> scal = colors.iterator();
        while (scal.hasNext()) {
            Scalar val = scal.next();
            ret.add(new Color((int) val.val[0], (int) val.val[1], (int) val.val[2]));
        }
        return ret;
    }


    // finding which color in the image is the "closest" color to either red or blue
    private static Color findClosestColor(Color targetColor, List<Color> colorList) {
        double minDistance = Double.MAX_VALUE;
        Color closestColor = null;

        for (Color color : colorList) {
            double distance = calculateColorDistance(targetColor, color);
            if (distance < minDistance) {
                minDistance = distance;
                closestColor = color;
            }
        }

        return closestColor;
    }


    // calculate euclidean distance in the  RGB color space between the colors.
    private static double calculateColorDistance(Color color1, Color color2) {
        int r1 = color1.getRed();
        int g1 = color1.getGreen();
        int b1 = color1.getBlue();

        int r2 = color2.getRed();
        int g2 = color2.getGreen();
        int b2 = color2.getBlue();

        // Calculate the Euclidean distance between two colors in RGB space
        return Math.sqrt(Math.pow(r1 - r2, 2) + Math.pow(g1 - g2, 2) + Math.pow(b1 - b2, 2));
    }
}
