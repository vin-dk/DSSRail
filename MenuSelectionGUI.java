import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import javafx.scene.chart.CategoryAxis;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.XYChart;

public class MenuSelectionGUI extends Application {

    private TextField[][] factorFields = new TextField[3][3]; // 3 categories, 3 factors each
    private Stage outputStage; // To track the output window

    @Override
    public void start(Stage primaryStage) {
        BorderPane root = new BorderPane();
        Scene scene = new Scene(root, 800, 600);  // Increased window size

        Label headerLabel = new Label("Menu Selection");
        headerLabel.setStyle("-fx-font-size: 18px;");
        root.setTop(headerLabel);
        BorderPane.setAlignment(headerLabel, Pos.CENTER);
        BorderPane.setMargin(headerLabel, new Insets(10, 0, 10, 0));

        VBox optionsBox = new VBox(20);
        optionsBox.setAlignment(Pos.CENTER_LEFT);
        optionsBox.setPadding(new Insets(10));

        // Rail Geometry section
        optionsBox.getChildren().add(createFactorGroup("Rail Geometry", 0, new int[]{5, 10, 20}));
        // Sleeper section
        optionsBox.getChildren().add(createFactorGroup("Sleeper", 1, new int[]{5, 10, 20}));
        // Rail section
        optionsBox.getChildren().add(createFactorGroup("Rail", 2, new int[]{5, 10, 20}));

        root.setCenter(optionsBox);

        // Enter button
        Button enterButton = new Button("Enter");
        enterButton.setOnAction(e -> {
            validateAndCompute(primaryStage);
        });

        BorderPane bottomPane = new BorderPane();
        bottomPane.setRight(enterButton);
        root.setBottom(bottomPane);
        BorderPane.setAlignment(bottomPane, Pos.BOTTOM_RIGHT);

        primaryStage.setScene(scene);
        primaryStage.setTitle("Menu Selection");
        primaryStage.show();
    }

    private VBox createFactorGroup(String label, int categoryIndex, int[] ranges) {
        VBox group = new VBox(10);
        group.setAlignment(Pos.CENTER_LEFT);

        Label categoryLabel = new Label(label);
        categoryLabel.setStyle("-fx-font-size: 14px;");
        group.getChildren().add(categoryLabel);

        GridPane gridPane = new GridPane();
        gridPane.setHgap(10);
        gridPane.setVgap(5);

        for (int i = 0; i < 3; i++) {
            Label factorLabel = new Label("Factor " + (i + 1) + " (" + "1-" + ranges[i] + "): ");
            TextField textField = new TextField();
            textField.setPromptText("Enter value");

            final int factorIndex = i;
            final int range = ranges[i];

            textField.focusedProperty().addListener((obs, oldVal, newVal) -> {
                if (!newVal) {
                    String input = textField.getText();
                    try {
                        int value = Integer.parseInt(input);
                        if (value < 1 || value > range) {
                            textField.clear();
                        }
                    } catch (NumberFormatException e) {
                        textField.clear();
                    }
                }
            });

            factorFields[categoryIndex][factorIndex] = textField;

            gridPane.add(factorLabel, 0, i);
            gridPane.add(textField, 1, i);
        }

        group.getChildren().add(gridPane);

        return group;
    }

    private void validateAndCompute(Stage primaryStage) {
        int[] factors = new int[9];
        int index = 0;
        boolean allValid = true;

        // Extracting the values from the 3 categories and 3 factors each
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                String input = factorFields[i][j].getText();
                if (input.isEmpty()) {
                    // Outline missing fields in red
                    factorFields[i][j].setStyle("-fx-border-color: red; -fx-border-width: 2px;");
                    allValid = false;
                } else {
                    // Clear red border if valid input
                    factorFields[i][j].setStyle("");
                    factors[index++] = Integer.parseInt(input);
                }
            }
        }

        if (allValid) {
            // Proceed with calculation and output
            double rawOutput = calculateOutput(factors[0], factors[1], factors[2], factors[3], factors[4], 
                                               factors[5], factors[6], factors[7], factors[8]);

            // Apply linear scaling to convert raw output to a rating between 1 and 5
            double rating = linearScale(rawOutput, 1, 5, 1, 425);
            String classification = classifyRating(rating);

            // Calculate costs for months 1, 6, and 12
            double costMonth1 = calculateCost(factors, 1);
            double costMonth6 = calculateCost(factors, 6);
            double costMonth12 = calculateCost(factors, 12);

            // Close previous output window, if it exists
            if (outputStage != null) {
                outputStage.close();
            }

            showOutputWindow(primaryStage, rawOutput, rating, classification, costMonth1, costMonth6, costMonth12);
            showBarGraph(primaryStage, factors);  // Show the new bar graph window
        }
    }

    private double calculateOutput(int a, int b, int c, int d, int e, int f, int g, int h, int i) {
        // Calculation using the provided formula
        return 9 * a + 5 * b + 2 * c + 8 * d + 2 * e + 3 * f + 2 * g + 4 * h + 6 * i;
    }

    private double calculateCost(int[] factors, int t) {
        // Cost calculation with additional term t^2
        return calculateOutput(factors[0], factors[1], factors[2], factors[3], factors[4], 
                               factors[5], factors[6], factors[7], factors[8]) + Math.pow(t, 2);
    }

    private double linearScale(double x, double newMin, double newMax, double oldMin, double oldMax) {
        // Apply linear scaling to map raw output to the desired range
        return ((x - oldMin) / (oldMax - oldMin)) * (newMax - newMin) + newMin;
    }

    private String classifyRating(double rating) {
        if (rating <= 2) {
            return "Perfect";
        } else if (rating < 4) {
            return "Fair";
        } else {
            return "Poor";
        }
    }

    private void showOutputWindow(Stage primaryStage, double rawOutput, double rating, String classification, 
                                  double costMonth1, double costMonth6, double costMonth12) {
        // Create a new window to display the output
        outputStage = new Stage();
        BorderPane outputRoot = new BorderPane();
        Scene outputScene = new Scene(outputRoot, 400, 300);

        Label headerLabel = new Label("Output");
        headerLabel.setStyle("-fx-font-size: 18px;");
        outputRoot.setTop(headerLabel);
        BorderPane.setAlignment(headerLabel, Pos.CENTER);
        BorderPane.setMargin(headerLabel, new Insets(10, 0, 10, 0));

        VBox resultBox = new VBox(10);
        resultBox.setAlignment(Pos.CENTER);
        Label outputLabel = new Label(String.format("Output: %.2f", rawOutput));
        Label ratingLabel = new Label(String.format("Rating: %.2f", rating));
        Label classificationLabel = new Label(String.format("Classification: %s", classification));
        Label costMonth1Label = new Label(String.format("Cost, Month 1: %.2f", costMonth1));
        Label costMonth6Label = new Label(String.format("Cost, Month 6: %.2f", costMonth6));
        Label costMonth12Label = new Label(String.format("Cost, Month 12: %.2f", costMonth12));

        resultBox.getChildren().addAll(outputLabel, ratingLabel, classificationLabel, costMonth1Label, costMonth6Label, costMonth12Label);
        outputRoot.setCenter(resultBox);

        outputStage.setScene(outputScene);
        outputStage.setTitle("Output");
        outputStage.show();
    }

    private void showBarGraph(Stage primaryStage, int[] factors) {
        // Create a new window for the bar graph
        Stage graphStage = new Stage();
        BorderPane graphRoot = new BorderPane();
        Scene graphScene = new Scene(graphRoot, 600, 400);

        // Create X and Y axes
        CategoryAxis xAxis = new CategoryAxis();
        xAxis.setLabel("Month");
        xAxis.getCategories().addAll("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12");

        // Calculate the maximum cost from month 0 to month 12
        double maxCost = 0;
        for (int t = 0; t <= 12; t++) {
            double cost = calculateCost(factors, t);
            if (cost > maxCost) {
                maxCost = cost;
            }
        }

        // Round the maxCost up to the next logical increment (e.g., 30, 60, 90, etc.)
        double yAxisUpperBound = Math.ceil(maxCost / 30) * 30;  // Round up to nearest 30

        // Create Y axis with dynamic upper bound
        NumberAxis yAxis = new NumberAxis(0, yAxisUpperBound, 30);
        yAxis.setLabel("Cost");

        // Create BarChart
        BarChart<String, Number> barChart = new BarChart<>(xAxis, yAxis);
        barChart.setTitle("Cost Over 12 Months");

        // Create a series for cost values
        XYChart.Series<String, Number> series = new XYChart.Series<>();
        series.setName("Cost");

        for (int t = 0; t <= 12; t++) {
            double cost = calculateCost(factors, t);
            series.getData().add(new XYChart.Data<>(String.valueOf(t), cost));
        }

        barChart.getData().add(series);
        graphRoot.setCenter(barChart);

        graphStage.setScene(graphScene);
        graphStage.setTitle("Cost Bar Graph");
        graphStage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}