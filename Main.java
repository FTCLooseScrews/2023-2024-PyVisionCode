import org.apache.commons.math3.ml.clustering.Cluster;
import org.apache.commons.math3.ml.clustering.Clusterer;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.markers.SeriesMarkers;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) throws IOException {
        String imagePath = "C:\\Users\\advay\\misc\\pic1.jpg";
        BufferedImage image = ImageIO.read(new File(imagePath));

        Image scaledImage = image.getScaledInstance(640, 480, Image.SCALE_DEFAULT);
        BufferedImage outputImage = new BufferedImage(640, 480, BufferedImage.TYPE_INT_RGB);
        outputImage.getGraphics().drawImage(scaledImage, 0, 0, null);

        BufferedImage normalizedImage = normalizeImage(outputImage);

        int numClusters = findOptimalNumClusters(normalizedImage);

        List<Color> dominantColors = extractDominantColors(normalizedImage, numClusters);

        for (Color color : dominantColors) {
            System.out.println("RGB: " + color.getRed() + ", " + color.getGreen() + ", " + color.getBlue());
            System.out.println("Hex: #" + Integer.toHexString(color.getRGB()).substring(2).toUpperCase());
            float[] hsv = rgbToHsv(color);
            System.out.println("HSV: " + hsv[0] + ", " + hsv[1] + ", " + hsv[2]);
            System.out.println("=====");
        }

        displayReducedColors(outputImage, dominantColors);
    }

    private static int findOptimalNumClusters(BufferedImage image) {
        List<DoublePoint> data = new ArrayList<>();
        int width = image.getWidth();
        int height = image.getHeight();

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int rgb = image.getRGB(x, y);
                int red = (rgb >> 16) & 0xFF;
                int green = (rgb >> 8) & 0xFF;
                int blue = rgb & 0xFF;

                data.add(new DoublePoint(new double[]{red, green, blue}));
            }
        }

        int maxClusters = 21;
        List<Double> costValues = new ArrayList<>();

        for (int numClusters = 1; numClusters <= maxClusters; numClusters++) {
            Clusterer<DoublePoint> clusterer = new KMeansPlusPlusClusterer<>(numClusters, -1, new EuclideanDistance());
            List<Cluster<DoublePoint>> clusters = (List<Cluster<DoublePoint>>) clusterer.cluster(data);

            // Calculate the cost (inertia)
            double cost = calculateInertia(clusters);
            costValues.add(cost);
        }

        // Find the optimal number of clusters using the elbow method
        int optimalNumClusters = findElbowPoint(costValues);

        saveElbowGraph(costValues);

        System.out.println("Optimal number of clusters: " + optimalNumClusters);
        return optimalNumClusters;
    }

    private static void saveElbowGraph(List<Double> costValues) {
        int maxClusters = costValues.size();
        int[] x = new int[maxClusters];
        double[] y = new double[maxClusters];

        for (int i = 0; i < maxClusters; i++) {
            x[i] = i + 1;
            y[i] = costValues.get(i);
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
            BitmapEncoder.saveBitmap(chart, "C:\\Users\\advay\\misc\\ElbowGraph.png", BitmapEncoder.BitmapFormat.PNG);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static double calculateInertia(List<Cluster<DoublePoint>> clusters) {
        double cost = 0.0;
        for (Cluster<DoublePoint> cluster : clusters) {
            DoublePoint centroid = calculateCentroid(cluster);
            List<DoublePoint> points = cluster.getPoints();

            for (DoublePoint point : points) {
                cost += Math.pow(distance(point, centroid), 2);
            }
        }
        return cost;
    }

    private static DoublePoint calculateCentroid(Cluster<DoublePoint> cluster) {
        List<DoublePoint> points = cluster.getPoints();
        int dimensions = points.get(0).getPoint().length;
        double[] sum = new double[dimensions];

        for (DoublePoint point : points) {
            double[] values = point.getPoint();
            for (int i = 0; i < dimensions; i++) {
                sum[i] += values[i];
            }
        }

        double[] centroid = new double[dimensions];
        for (int i = 0; i < dimensions; i++) {
            centroid[i] = sum[i] / points.size();
        }

        return new DoublePoint(centroid);
    }

    private static double distance(DoublePoint point1, DoublePoint point2) {
        double[] values1 = point1.getPoint();
        double[] values2 = point2.getPoint();

        double sum = 0.0;
        for (int i = 0; i < values1.length; i++) {
            sum += Math.pow(values1[i] - values2[i], 2);
        }

        return Math.sqrt(sum);
    }

    private static int findElbowPoint(List<Double> costValues) {
        int optimalNumClusters = 1;
        double prevCost = costValues.get(1);
        double maxChange = 0.0;
        double minAngle = 0;
        int kVal = 1;


        for (int i = 2; i < costValues.size()-1; i++) {
            System.out.println("K-Value: " + (i+1) + ", Cost Value: " + costValues.get(i));
            if (costValues.get(i) > costValues.get(i - 1)) continue;
            double angle = findAngle(1, (costValues.get(i) - costValues.get(i-1))/1E8, 1, (costValues.get(i+1) - costValues.get(i))/1E8);
            System.out.println("Degree measure at " + (i+1) + ": " + angle);
            if (Math.abs(Math.PI/2 - angle) < Math.abs(Math.PI / 2 - minAngle)) {
                minAngle = angle;
                kVal = i+1;
            }

//            double cost = costValues.get(i);
//            double change = prevCost - cost;
//
//            if (change > maxChange) {
//                maxChange = change;
//                optimalNumClusters = i + 1;
//            }
//
//            prevCost = cost;
        }

//        return optimalNumClusters;
        return kVal;
    }


    private static List<Color> extractDominantColors(BufferedImage image, int numClusters) {
        List<DoublePoint> data = new ArrayList<>();

        int width = image.getWidth();
        int height = image.getHeight();

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int rgb = image.getRGB(x, y);
                int red = (rgb >> 16) & 0xFF;
                int green = (rgb >> 8) & 0xFF;
                int blue = rgb & 0xFF;

                data.add(new DoublePoint(new double[]{red, green, blue}));
            }
        }

        // Use K-Means clustering to group similar colors
        Clusterer<DoublePoint> clusterer = new KMeansPlusPlusClusterer<>(numClusters, -1, new EuclideanDistance());
        List<Cluster<DoublePoint>> clusters = (List<Cluster<DoublePoint>>) clusterer.cluster(data);

        List<Color> dominantColors = new ArrayList<>();

        // Calculate the centroid (mean) of each cluster
        for (Cluster<DoublePoint> cluster : clusters) {
            List<DoublePoint> points = cluster.getPoints();
            double sumRed = 0.0, sumGreen = 0.0, sumBlue = 0.0;

            for (DoublePoint point : points) {
                sumRed += point.getPoint()[0];
                sumGreen += point.getPoint()[1];
                sumBlue += point.getPoint()[2];
            }

            int clusterSize = points.size();
            int red = (int) (sumRed / clusterSize);
            int green = (int) (sumGreen / clusterSize);
            int blue = (int) (sumBlue / clusterSize);

            dominantColors.add(new Color(red, green, blue));
        }

        return dominantColors;
    }

    private static BufferedImage normalizeImage(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();

        BufferedImage normalizedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int rgb = image.getRGB(x, y);
                int red = (rgb >> 16) & 0xFF;
                int green = (rgb >> 8) & 0xFF;
                int blue = rgb & 0xFF;

                float normRed = red / 255.0f;
                float normGreen = green / 255.0f;
                float normBlue = blue / 255.0f;

                int normalizedRGB = new Color(normRed, normGreen, normBlue).getRGB();
                normalizedImage.setRGB(x, y, normalizedRGB);
            }
        }

        return normalizedImage;
    }

    private static float[] rgbToHsv(Color color) {
        float[] hsv = new float[3];
        Color.RGBtoHSB(color.getRed(), color.getGreen(), color.getBlue(), hsv);
        return hsv;
    }

    private static void displayReducedColors(BufferedImage image, List<Color> dominantColors) {
        BufferedImage reducedImage = createReducedColorsImage(image, dominantColors);

        try {
            File outputFile = new File("C:\\Users\\advay\\misc\\reduced_colors.jpg");
            ImageIO.write(reducedImage, "jpg", outputFile);
            System.out.println("Reduced colors image saved to: " + outputFile.getAbsolutePath());
        } catch (Exception ignored) {}
    }

    private static BufferedImage createReducedColorsImage(BufferedImage originalImage, List<Color> dominantColors) {
        int width = originalImage.getWidth();
        int height = originalImage.getHeight();

        BufferedImage reducedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int rgb = originalImage.getRGB(x, y);
                Color originalColor = new Color(rgb);
                Color closestDominantColor = findClosestColor(originalColor, dominantColors);
                reducedImage.setRGB(x, y, closestDominantColor.getRGB());
            }
        }

        return reducedImage;
    }

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

    public static double findAngle(double x1, double y1, double x2, double y2) {
        double dotProduct = x1 * x2 + y1 * y2;
        double magAB = Math.sqrt(Math.pow(x1, 2) + Math.pow(y1, 2)) * Math.sqrt(Math.pow(x2, 2) + Math.pow(y2, 2));
        return Math.acos(dotProduct/magAB);
    }

}