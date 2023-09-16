import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.markers.SeriesMarkers;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;

public class MatMain {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat image = Imgcodecs.imread("C:\\Users\\advay\\misc\\pic1.jpg");
        Imgproc.resize(image, image, new Size(640, 480));

        Mat blurredImage = new Mat();
        Imgproc.GaussianBlur(image, blurredImage, new Size(15, 15),0, 0);

        Imgcodecs.imwrite("C:\\Users\\advay\\misc\\output.jpg", applyKmeans(blurredImage, findOptimalK(blurredImage)));
        Imgcodecs.imwrite("C:\\Users\\advay\\misc\\blurred.jpg", blurredImage);
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

    public static double findAngle(double x1, double y1, double x2, double y2) {
        double dotProduct = x1 * x2 + y1 * y2;
        double magAB = Math.sqrt(Math.pow(x1, 2) + Math.pow(y1, 2)) * Math.sqrt(Math.pow(x2, 2) + Math.pow(y2, 2));
        return Math.acos(dotProduct/magAB);
    }

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
            BitmapEncoder.saveBitmap(chart, "C:\\Users\\advay\\misc\\ElbowGraphOpenCV.png", BitmapEncoder.BitmapFormat.PNG);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static Mat applyKmeans(Mat image, int optimalK) {
        image.convertTo(image, CvType.CV_32F);
        Mat data = image.reshape(1, (int)image.total());

        Mat bestLabels = new Mat();
        TermCriteria criteria = new TermCriteria();
        int attempts=5;
        int flags = Core.KMEANS_RANDOM_CENTERS;
        Mat centers = new Mat();

        int K = optimalK;

        Core.kmeans(data, K, bestLabels, criteria, attempts, flags, centers);

        Mat draw = new Mat((int)image.total(),1, CvType.CV_32FC3);
        Mat colors = centers.reshape(3,K);
        for (int i=0; i<K; i++) {
            Mat mask = new Mat(); // a mask for each cluster label
            Core.compare(bestLabels, new Scalar(i), mask, Core.CMP_EQ);
            Mat col = colors.row(i); // can't use the Mat directly with setTo() (see #19100)
            double d[] = col.get(0,0); // can't create Scalar directly from get(), 3 vs 4 elements
            draw.setTo(new Scalar(d[0],d[1],d[2]), mask);
        }

        draw = draw.reshape(3, image.rows());
        draw.convertTo(draw, CvType.CV_8U);

        return draw;
    }
}
