import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.RadioButton;
import javafx.scene.control.TextField;
import javafx.scene.control.ToggleGroup;
import javafx.scene.input.KeyCode;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

public class RailGeometry {

    public float tgiOut = 0;
    public String condition = "";
    public String recommendedCourse = "";
    private Stage resultStage;

    public String[] assignments = {"Very Poor", "Poor", "Fair", "Good", "Excellent"};
    public String[] recommendation = {"Immediate shutdown or major repairs required", "Urgent repairs necessary; restrict speeds",
            "Immediate corrective actions planned", "Schedule preventive maintenance", "Routine monitoring, no immediate action required"};
   

    private String userChoice = "None";
    
    public static double stdv(float[] values) { // helper method for calculating standard deviation 
        int n = values.length;
        if (n == 0) {
            throw new IllegalArgumentException("Array must have at least one element.");
        }
        
        double mean = 0;
        for (float value : values) {
            mean += value;
        }
        mean /= n;
        
        
        double sumSquaredDiffs = 0;
        for (float value : values) {
            sumSquaredDiffs += Math.pow(value - mean, 2);
        }

        double stdDev = Math.sqrt(sumSquaredDiffs / (n - 1));

        return stdDev;
    }
    
 // method to calculate the average standard deviation (σH)(assumes implementation as described) 
    public static double genH(float[] HLEFT, float[] HRIGHT) {
        double stdDevLeft = stdv(HLEFT);

        double stdDevRight = stdv(HRIGHT);

        double sigmaH = (stdDevLeft + stdDevRight) / 2;

        return sigmaH;
    }
    
    public static double genS(float[] crossLevel, float[] gauge, float[] horizontalDeviation) { // method calculating stdv for s
        if (crossLevel.length != gauge.length || crossLevel.length != horizontalDeviation.length) {
            throw new IllegalArgumentException("All arrays must be of the same size.");
        }

        float[] Si = new float[crossLevel.length];
        for (int i = 0; i < crossLevel.length; i++) {
            Si[i] = crossLevel[i] * gauge[i] * horizontalDeviation[i];
        }

        return stdv(Si);
    }

    public void openMenuSelection(Stage primaryStage) {
        BorderPane root = new BorderPane();
        root.setPadding(new Insets(10));

        VBox optionsBox = new VBox(10);
        optionsBox.setPadding(new Insets(10));
        optionsBox.setAlignment(Pos.CENTER_LEFT);

        Label instructionLabel = new Label("Please select an option:");
        optionsBox.getChildren().add(instructionLabel);

        ToggleGroup toggleGroup = new ToggleGroup();

        RadioButton defaultOption = new RadioButton("Default");
        defaultOption.setToggleGroup(toggleGroup);
        defaultOption.setOnAction(e -> userChoice = "Default");

        for (int i = 1; i <= 5; i++) {
            final int variantNumber = i;
            RadioButton variantOption = new RadioButton("Variant " + variantNumber);
            variantOption.setToggleGroup(toggleGroup);
            variantOption.setOnAction(e -> userChoice = "Variant " + variantNumber);
            optionsBox.getChildren().add(variantOption);
        }

        optionsBox.getChildren().add(defaultOption);

        root.setLeft(optionsBox);

        Button enterButton = new Button("Enter");
        enterButton.setOnAction(e -> {
            System.out.println("User Choice: " + userChoice);
            if ("Default".equals(userChoice)) {
                openDefaultWindow();
                }
            else if ("Variant 1".equals(userChoice)) {
            	openVarOneWindow();
            }
            else if ("Variant 2".equals(userChoice)) {
            	openVarTwoWindow();
            }
            else if ("Variant 3".equals(userChoice)) {
            	openVarThreeWindow();
            }
            else if ("Variant 4".equals(userChoice)) {
            	openVarFourWindow();
            }
            else if ("Variant 5".equals(userChoice)) {
            	openVarFiveWindow();
            }
            else {
                primaryStage.close();
            }
        });

        BorderPane bottomPane = new BorderPane();
        bottomPane.setRight(enterButton);
        BorderPane.setMargin(enterButton, new Insets(10));
        root.setBottom(bottomPane);

        Scene scene = new Scene(root, 300, 250);
        primaryStage.setScene(scene);
        primaryStage.setTitle("Menu Selection");
    }

    private void openDefaultWindow() {
        Stage defaultWindow = new Stage();
        BorderPane pane = new BorderPane();
        pane.setPadding(new Insets(10));

        GridPane gridPane = new GridPane();
        gridPane.setPadding(new Insets(10));
        gridPane.setHgap(10);
        gridPane.setVgap(10);
        gridPane.setAlignment(Pos.CENTER_LEFT);

        Label notice = new Label("All measurements in mm");
        notice.setStyle("-fx-font-weight: bold;");

        Label longitudinalLabel = new Label("Longitudinal Deviation: ");
        TextField longitudinalInput = new TextField();
        configureInputField(longitudinalInput);

        Label alignmentLabel = new Label("Alignment Deviation: ");
        TextField alignmentInput = new TextField();
        configureInputField(alignmentInput);

        Label gaugeLabel = new Label("Gauge Deviation: ");
        TextField gaugeInput = new TextField();
        configureInputField(gaugeInput);

        Label allowanceLLabel = new Label("Allowance Longitudinal: ");
        TextField allowanceLInput = new TextField();
        configurePositiveInputField(allowanceLInput);

        Label allowanceALabel = new Label("Allowance Alignment: ");
        TextField allowanceAInput = new TextField();
        configurePositiveInputField(allowanceAInput);

        Label allowanceGLabel = new Label("Allowance Gauge: ");
        TextField allowanceGInput = new TextField();
        configurePositiveInputField(allowanceGInput);

        longitudinalInput.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.ENTER) {
                alignmentInput.requestFocus();
            }
        });

        alignmentInput.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.ENTER) {
                gaugeInput.requestFocus();
            }
        });

        gaugeInput.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.ENTER) {
                allowanceLInput.requestFocus();
            }
        });

        allowanceLInput.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.ENTER) {
                allowanceAInput.requestFocus();
            }
        });

        allowanceAInput.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.ENTER) {
                allowanceGInput.requestFocus();
            }
        });

        gridPane.add(notice, 0, 0);
        gridPane.add(longitudinalLabel, 0, 1);
        gridPane.add(longitudinalInput, 1, 1);
        gridPane.add(alignmentLabel, 0, 2);
        gridPane.add(alignmentInput, 1, 2);
        gridPane.add(gaugeLabel, 0, 3);
        gridPane.add(gaugeInput, 1, 3);
        gridPane.add(allowanceLLabel, 0, 4);
        gridPane.add(allowanceLInput, 1, 4);
        gridPane.add(allowanceALabel, 0, 5);
        gridPane.add(allowanceAInput, 1, 5);
        gridPane.add(allowanceGLabel, 0, 6);
        gridPane.add(allowanceGInput, 1, 6);

        pane.setCenter(gridPane);

        Button enterButton = new Button("Enter");
        enterButton.setOnAction(e -> {
            float l = Float.parseFloat(longitudinalInput.getText());
            float a = Float.parseFloat(alignmentInput.getText());
            float g = Float.parseFloat(gaugeInput.getText());
            float allowanceL = Float.parseFloat(allowanceLInput.getText());
            float allowanceA = Float.parseFloat(allowanceAInput.getText());
            float allowanceG = Float.parseFloat(allowanceGInput.getText());

            defaultTGI(l, a, g, allowanceL, allowanceA, allowanceG);
            defaultWindow.close();
        });

        BorderPane bottomPane = new BorderPane();
        bottomPane.setRight(enterButton);
        BorderPane.setMargin(enterButton, new Insets(10));
        pane.setBottom(bottomPane);

        Scene scene = new Scene(pane, 400, 300);
        defaultWindow.setScene(scene);
        defaultWindow.setTitle("Default Deviation Input");
        defaultWindow.show();
    }
    
    private void openVarOneWindow() {
        Stage VarOneWindow = new Stage();
        BorderPane pane = new BorderPane();
        pane.setPadding(new Insets(10));

        GridPane gridPane = new GridPane();
        gridPane.setPadding(new Insets(10));
        gridPane.setHgap(10);
        gridPane.setVgap(10);
        gridPane.setAlignment(Pos.CENTER_LEFT);

        Label notice = new Label("All measurements in mm");
        notice.setStyle("-fx-font-weight: bold;");

        Label longitudinalLabel = new Label("Longitudinal Deviation: ");
        TextField longitudinalInput = new TextField();
        configureInputField(longitudinalInput);

        Label alignmentLabel = new Label("Alignment Deviation: ");
        TextField alignmentInput = new TextField();
        configureInputField(alignmentInput);

        Label gaugeLabel = new Label("Gauge Deviation: ");
        TextField gaugeInput = new TextField();
        configureInputField(gaugeInput);

        Label WLLabel = new Label("WL: ");
        TextField WLInput = new TextField();
        configurePositiveInputField(WLInput);
        configureBoundedInputField(WLInput); 

        Label WALabel = new Label("WA: ");
        TextField WAInput = new TextField();
        configurePositiveInputField(WAInput);
        configureBoundedInputField(WAInput); 

        Label WGLabel = new Label("WG: ");
        TextField WGInput = new TextField();
        configurePositiveInputField(WGInput);
        configureBoundedInputField(WGInput); 

        
        longitudinalInput.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.ENTER) {
                alignmentInput.requestFocus();
            }
        });

        alignmentInput.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.ENTER) {
                gaugeInput.requestFocus();
            }
        });

        gaugeInput.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.ENTER) {
                WLInput.requestFocus();
            }
        });

        WLInput.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.ENTER) {
                WAInput.requestFocus();
            }
        });

        WAInput.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.ENTER) {
                WGInput.requestFocus();
            }
        });

        gridPane.add(notice, 0, 0);
        gridPane.add(longitudinalLabel, 0, 1);
        gridPane.add(longitudinalInput, 1, 1);
        gridPane.add(alignmentLabel, 0, 2);
        gridPane.add(alignmentInput, 1, 2);
        gridPane.add(gaugeLabel, 0, 3);
        gridPane.add(gaugeInput, 1, 3);
        gridPane.add(WLLabel, 0, 4);
        gridPane.add(WLInput, 1, 4);
        gridPane.add(WALabel, 0, 5);
        gridPane.add(WAInput, 1, 5);
        gridPane.add(WGLabel, 0, 6);
        gridPane.add(WGInput, 1, 6);

        pane.setCenter(gridPane);

        Button enterButton = new Button("Enter");
        enterButton.setOnAction(e -> {
            float l = Float.parseFloat(longitudinalInput.getText());
            float a = Float.parseFloat(alignmentInput.getText());
            float g = Float.parseFloat(gaugeInput.getText());
            float WL = Float.parseFloat(WLInput.getText());
            float WA = Float.parseFloat(WAInput.getText());
            float WG = Float.parseFloat(WGInput.getText());

            varTGIone(l, a, g, WL, WA, WG);
            VarOneWindow.close();
        });

        BorderPane bottomPane = new BorderPane();
        bottomPane.setRight(enterButton);
        BorderPane.setMargin(enterButton, new Insets(10));
        pane.setBottom(bottomPane);

        Scene scene = new Scene(pane, 400, 300);
        VarOneWindow.setScene(scene);
        VarOneWindow.setTitle("Variation 1 Deviation Input");
        VarOneWindow.show();
    }
    
    private void openVarTwoWindow() {
        Stage VarTwoWindow = new Stage();
        BorderPane pane = new BorderPane();
        pane.setPadding(new Insets(10));

        GridPane gridPane = new GridPane();
        gridPane.setPadding(new Insets(10));
        gridPane.setHgap(10);
        gridPane.setVgap(10);
        gridPane.setAlignment(Pos.CENTER_LEFT);

        Label notice = new Label("All measurements in mm");
        notice.setStyle("-fx-font-weight: bold;");

        Label longitudinalLabel = new Label("Longitudinal Deviation: ");
        TextField longitudinalInput = new TextField();
        configureInputField(longitudinalInput);
        configureBoundedInputField(longitudinalInput);

        Label alignmentLabel = new Label("Alignment Deviation: ");
        TextField alignmentInput = new TextField();
        configureInputField(alignmentInput);
        configureBoundedInputField(alignmentInput);

        Label gaugeLabel = new Label("Gauge Deviation: ");
        TextField gaugeInput = new TextField();
        configureInputField(gaugeInput);
        configureBoundedInputField(gaugeInput);

        longitudinalInput.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.ENTER) {
                alignmentInput.requestFocus();
            }
        });

        alignmentInput.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.ENTER) {
                gaugeInput.requestFocus();
            }
        });

        gridPane.add(notice, 0, 0);
        gridPane.add(longitudinalLabel, 0, 1);
        gridPane.add(longitudinalInput, 1, 1);
        gridPane.add(alignmentLabel, 0, 2);
        gridPane.add(alignmentInput, 1, 2);
        gridPane.add(gaugeLabel, 0, 3);
        gridPane.add(gaugeInput, 1, 3);
        
        pane.setCenter(gridPane);

        Button enterButton = new Button("Enter");
        enterButton.setOnAction(e -> {
            float l = Float.parseFloat(longitudinalInput.getText());
            float a = Float.parseFloat(alignmentInput.getText());
            float g = Float.parseFloat(gaugeInput.getText());

            varTGItwo(l, a, g);
            VarTwoWindow.close();
        });

        BorderPane bottomPane = new BorderPane();
        bottomPane.setRight(enterButton);
        BorderPane.setMargin(enterButton, new Insets(10));
        pane.setBottom(bottomPane);

        Scene scene = new Scene(pane, 400, 300);
        VarTwoWindow.setScene(scene);
        VarTwoWindow.setTitle("Variation 2 Deviation Input");
        VarTwoWindow.show();
    }
    
    private void openVarThreeWindow() {
        Stage VarTwoWindow = new Stage();
        BorderPane pane = new BorderPane();
        pane.setPadding(new Insets(10));

        GridPane gridPane = new GridPane();
        gridPane.setPadding(new Insets(10));
        gridPane.setHgap(10);
        gridPane.setVgap(10);
        gridPane.setAlignment(Pos.CENTER_LEFT);

        Label notice = new Label("All measurements in mm");
        notice.setStyle("-fx-font-weight: bold;");

        Label stdLabel = new Label("Standard Deviation: ");
        TextField stdInput = new TextField();
        configureInputField(stdInput);

        Label stdvLabel = new Label("80th Percentile Deviation: ");
        TextField stdvInput = new TextField();
        configureInputField(stdvInput);

        stdInput.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.ENTER) {
                stdvInput.requestFocus();
            }
        });

        stdvInput.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.ENTER) {
                stdvInput.requestFocus();
            }
        });

        gridPane.add(notice, 0, 0);
        gridPane.add(stdLabel, 0, 1);
        gridPane.add(stdInput, 1, 1);
        gridPane.add(stdvLabel, 0, 2);
        gridPane.add(stdvInput, 1, 2);
        
        pane.setCenter(gridPane);

        Button enterButton = new Button("Enter");
        enterButton.setOnAction(e -> {
            float l = Float.parseFloat(stdInput.getText());
            float a = Float.parseFloat(stdvInput.getText());

            varTGIthree(l, a);
            VarTwoWindow.close();
        });

        BorderPane bottomPane = new BorderPane();
        bottomPane.setRight(enterButton);
        BorderPane.setMargin(enterButton, new Insets(10));
        pane.setBottom(bottomPane);

        Scene scene = new Scene(pane, 400, 300);
        VarTwoWindow.setScene(scene);
        VarTwoWindow.setTitle("Variation 3 Input");
        VarTwoWindow.show();
    }
    
    private void openVarFourWindow() {
        Stage VarTwoWindow = new Stage();
        BorderPane pane = new BorderPane();
        pane.setPadding(new Insets(10));

        GridPane gridPane = new GridPane();
        gridPane.setPadding(new Insets(10));
        gridPane.setHgap(10);
        gridPane.setVgap(10);
        gridPane.setAlignment(Pos.CENTER_LEFT);

        Label notice = new Label("All measurements in mm");
        notice.setStyle("-fx-font-weight: bold;");

        Label stdLabel = new Label("∑l (longitudinal): ");
        TextField stdInput = new TextField();
        configureInputField(stdInput);

        Label stdvLabel = new Label("L (longitudinal): ");
        TextField stdvInput = new TextField();
        configureInputField(stdvInput);
        
        Label std2Label = new Label("∑l (alignment): ");
        TextField std2Input = new TextField();
        configureInputField(std2Input);

        Label stdv2Label = new Label("L (alignment): ");
        TextField stdv2Input = new TextField();
        configureInputField(stdv2Input);
        
        Label std3Label = new Label("∑l (gauge): ");
        TextField std3Input = new TextField();
        configureInputField(std3Input);

        Label stdv3Label = new Label("L (gauge): ");
        TextField stdv3Input = new TextField();
        configureInputField(stdv3Input);

        stdInput.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.ENTER) {
                stdvInput.requestFocus();
            }
        });

        stdvInput.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.ENTER) {
                std2Input.requestFocus();
            }
        });
        
        std2Input.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.ENTER) {
                stdv2Input.requestFocus();
            }
        });
        
        stdv2Input.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.ENTER) {
                std3Input.requestFocus();
            }
        });
        
        std3Input.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.ENTER) {
                stdv3Input.requestFocus();
            }
        });
        
        stdv3Input.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.ENTER) {
                stdv3Input.requestFocus();
            }
        });

        gridPane.add(notice, 0, 0);
        gridPane.add(stdLabel, 0, 1);
        gridPane.add(stdInput, 1, 1);
        gridPane.add(stdvLabel, 0, 2);
        gridPane.add(stdvInput, 1, 2);
        gridPane.add(std2Label, 0, 3);
        gridPane.add(std2Input, 1, 3);
        gridPane.add(stdv2Label, 0, 4);
        gridPane.add(stdv2Input, 1, 4);
        gridPane.add(std3Label, 0, 5);
        gridPane.add(std3Input, 1, 5);
        gridPane.add(stdv3Label, 0, 6);
        gridPane.add(stdv3Input, 1, 6);
        
        pane.setCenter(gridPane);

        Button enterButton = new Button("Enter");
        enterButton.setOnAction(e -> {
            float l = Float.parseFloat(stdInput.getText());
            float a = Float.parseFloat(stdvInput.getText());
            float l2 = Float.parseFloat(std2Input.getText());
            float a2 = Float.parseFloat(stdv2Input.getText());
            float l3 = Float.parseFloat(std3Input.getText());
            float a3 = Float.parseFloat(stdv3Input.getText());

            varTGIfour(l, a, l2, a2, l3, a3);
            VarTwoWindow.close();
        });

        BorderPane bottomPane = new BorderPane();
        bottomPane.setRight(enterButton);
        BorderPane.setMargin(enterButton, new Insets(10));
        pane.setBottom(bottomPane);

        Scene scene = new Scene(pane, 400, 300);
        VarTwoWindow.setScene(scene);
        VarTwoWindow.setTitle("Variation 4 Input");
        VarTwoWindow.show();
    } 
    
    private void openVarFiveWindow() {
        Stage varFiveWindow = new Stage();
        BorderPane pane = new BorderPane();
        pane.setPadding(new Insets(10));

        GridPane gridPane = new GridPane();
        gridPane.setPadding(new Insets(10));
        gridPane.setHgap(10);
        gridPane.setVgap(10);
        gridPane.setAlignment(Pos.CENTER_LEFT);

        Label notice = new Label("All measurements in mm");
        notice.setStyle("-fx-font-weight: bold;");

        // Input fields for HLeft, HRight, Cross Levels, Gauges, and Horizontal Deviations
        Label hLeftLabel = new Label("H Left (Comma Separated list): ");
        TextField hLeftInput = new TextField();
        
        Label hRightLabel = new Label("H Right (Comma Separated list): ");
        TextField hRightInput = new TextField();

        Label crossLevelsLabel = new Label("Cross Levels (Comma Separated list): ");
        TextField crossLevelsInput = new TextField();

        Label gaugesLabel = new Label("Gauges (Comma Separated list): ");
        TextField gaugesInput = new TextField();

        Label horizontalDeviationsLabel = new Label("Horizontal Deviations (Comma Separated list): ");
        TextField horizontalDeviationsInput = new TextField();
        
        Label HLimLabel = new Label("σ_HLim: ");
        TextField HLimInput = new TextField();
        configureInputField(HLimInput);
        
        Label sLimLabel = new Label("σ_sLim: ");
        TextField sLimInput = new TextField();
        configurePositiveInputField(sLimInput); 

        // Add labels and inputs to the grid pane
        gridPane.add(notice, 0, 0);
        gridPane.add(hLeftLabel, 0, 1);
        gridPane.add(hLeftInput, 1, 1);
        gridPane.add(hRightLabel, 0, 2);
        gridPane.add(hRightInput, 1, 2);
        gridPane.add(crossLevelsLabel, 0, 3);
        gridPane.add(crossLevelsInput, 1, 3);
        gridPane.add(gaugesLabel, 0, 4);
        gridPane.add(gaugesInput, 1, 4);
        gridPane.add(horizontalDeviationsLabel, 0, 5);
        gridPane.add(horizontalDeviationsInput, 1, 5);
        gridPane.add(HLimLabel, 0, 6);
        gridPane.add(HLimInput, 1, 6);
        gridPane.add(sLimLabel, 0, 7);
        gridPane.add(sLimInput, 1, 7);

        pane.setCenter(gridPane);

        Button enterButton = new Button("Enter");
        enterButton.setOnAction(e -> {
            try {
                // Parse inputs to float arrays
                float[] HLEFT = parseInputToFloatArray(hLeftInput.getText());
                float[] HRIGHT = parseInputToFloatArray(hRightInput.getText());
                float[] crossLevels = parseInputToFloatArray(crossLevelsInput.getText());
                float[] gauges = parseInputToFloatArray(gaugesInput.getText());
                float[] horizontalDeviations = parseInputToFloatArray(horizontalDeviationsInput.getText());

                // Calculate σH and σs
                double sigmaH = genH(HLEFT, HRIGHT);
                double sigmaS = genS(crossLevels, gauges, horizontalDeviations);

                // Assuming you have σ_HLim and σ_sLim inputs (add these fields as necessary)
                float sigmaHLim = Float.parseFloat(HLimInput.getText()); // Replace with actual input field for σ_HLim
                float sigmaSLim = Float.parseFloat(sLimInput.getText()); // Replace with actual input field for σ_sLim

                // Call varTGIfive with the calculated values
                varTGIfive((float) sigmaH, sigmaHLim, (float) sigmaS, sigmaSLim);
                varFiveWindow.close();
            } catch (NumberFormatException ex) {
                // Handle number format exceptions or any other parsing errors
                showError("Please enter valid numerical values.");
            } catch (IllegalArgumentException ex) {
                // Handle cases where input arrays are of different sizes
                showError(ex.getMessage());
            }
        });

        BorderPane bottomPane = new BorderPane();
        bottomPane.setRight(enterButton);
        BorderPane.setMargin(enterButton, new Insets(10));
        pane.setBottom(bottomPane);

        Scene scene = new Scene(pane, 400, 400);
        varFiveWindow.setScene(scene);
        varFiveWindow.setTitle("Variation 5 Deviation Input");
        varFiveWindow.show();
    }

    private float[] parseInputToFloatArray(String input) {
        // helper function 
        String[] stringValues = input.trim().split("\\s*,\\s*");
        float[] floatValues = new float[stringValues.length];

        for (int i = 0; i < stringValues.length; i++) {
            floatValues[i] = Float.parseFloat(stringValues[i]);
        }

        return floatValues;
    }

    private void showError(String message) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle("Input Error");
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }

    private void configureBoundedInputField(TextField textField) {
        textField.textProperty().addListener((observable, oldValue, newValue) -> {
            try {
                if (newValue.isEmpty()) {
                    return;
                }
                float value = Float.parseFloat(newValue);
                if (value < 0.0 || value > 1.0) {
                    textField.setText(oldValue); 
                }
            } catch (NumberFormatException e) {
                textField.setText(oldValue); 
            }
        });
    }
    
    private void configureInputField(TextField textField) {
        textField.setPromptText("Enter a positive number");
    }

    private void configurePositiveInputField(TextField textField) {
        textField.setPromptText("Enter a positive number");
    }

    private void defaultTGI(float l, float a, float g, float allowanceL, float allowanceA, float allowanceG) {
        float exceedL = l - allowanceL > 0 ? l - allowanceL : 0;
        float exceedA = a - allowanceA > 0 ? a - allowanceA : 0;
        float exceedG = g - allowanceG > 0 ? g - allowanceG : 0;

        tgiOut = 100 - ((l + a + g) / (allowanceL + allowanceA + allowanceG)) * 100;
        tgiOut = Math.round(tgiOut);
        if (tgiOut < 0) {
            condition = "Very Poor";
            recommendedCourse = "Immediate shutdown or major repairs required";
        } else {
        condition = assignments[(int) Math.min(Math.max(tgiOut / 20, 0), 4)];
        recommendedCourse = recommendation[Math.min((int) tgiOut / 20, 4)];
        }

        resultStage = new Stage();
        BorderPane resultPane = new BorderPane();
        resultPane.setPadding(new Insets(10));

        VBox resultBox = new VBox(10);
        resultBox.setPadding(new Insets(10));
        resultBox.setAlignment(Pos.CENTER_LEFT);

        resultBox.getChildren().add(new Label("TGI Output: " + tgiOut));
        resultBox.getChildren().add(new Label("Condition: " + condition));
        resultBox.getChildren().add(new Label("Recommended Course of Action: " + recommendedCourse));
        resultBox.getChildren().add(new Label("Longitudinal exceedance: " + exceedL));
        resultBox.getChildren().add(new Label("Alignment exceedance: " + exceedA));
        resultBox.getChildren().add(new Label("Gauge exceedance: " + exceedG));

        resultPane.setCenter(resultBox);

        Scene resultScene = new Scene(resultPane, 400, 250);
        resultStage.setScene(resultScene);
        resultStage.setTitle("Results");
        resultStage.show();
    }
    
    private void varTGIone (float l, float a, float g, float WL, float WA, float WG) {
    	float num = ((WL * l) + (WA * a) + (WG * g));
    	float factor = num/10; 
    	float detract = factor * 100;
    	tgiOut = 100 - detract;
    	tgiOut = Math.round(tgiOut);
    	if (tgiOut < 0) {
            condition = "Very Poor";
            recommendedCourse = "Immediate shutdown or major repairs required";
        } else {
        condition = assignments[(int) Math.min(Math.max(tgiOut / 20, 0), 4)];
        recommendedCourse = recommendation[Math.min((int) tgiOut / 20, 4)];
        }

        resultStage = new Stage();
        BorderPane resultPane = new BorderPane();
        resultPane.setPadding(new Insets(10));

        VBox resultBox = new VBox(10);
        resultBox.setPadding(new Insets(10));
        resultBox.setAlignment(Pos.CENTER_LEFT);

        resultBox.getChildren().add(new Label("TGI Output: " + tgiOut));
        resultBox.getChildren().add(new Label("Condition: " + condition));
        resultBox.getChildren().add(new Label("Recommended Course of Action: " + recommendedCourse));

        resultPane.setCenter(resultBox);

        Scene resultScene = new Scene(resultPane, 400, 250);
        resultStage.setScene(resultScene);
        resultStage.setTitle("Results");
        resultStage.show();
    }
    
    private void varTGItwo (float l, float a, float g) {
    	tgiOut = (float) Math.sqrt(l * l + a * a + g * g);
    	if (tgiOut < 0) {
            condition = "Very Poor";
            recommendedCourse = "Urgent repairs needed";
        }
    	else if (tgiOut >= 0.0 && tgiOut < 0.4 )  {
    		condition = "Very Poor Quality";
    		recommendedCourse = "Urgent repairs needed";
        }
    	else if (tgiOut >= 0.4 && tgiOut < 0.6 )  {
    		condition = "Poor Quality";
    		recommendedCourse = "Immediate corrective actions required";
        }
    	else if (tgiOut >= 0.6 && tgiOut < 0.8 )  {
    		condition = "Fair Quality";
    		recommendedCourse = "Planned maintenance needed";
        }
    	else if (tgiOut >= 0.8 && tgiOut <= 1.0 )  {
    		condition = "Good Quality";
    		recommendedCourse = "Routine Monitoring";
        }
    	else {
    		condition = "Good Quality";
    		recommendedCourse = "Routine Monitoring";
    		System.out.print("System Error");
    	}

        resultStage = new Stage();
        BorderPane resultPane = new BorderPane();
        resultPane.setPadding(new Insets(10));

        VBox resultBox = new VBox(10);
        resultBox.setPadding(new Insets(10));
        resultBox.setAlignment(Pos.CENTER_LEFT);

        resultBox.getChildren().add(new Label("TGI Output: " + tgiOut));
        resultBox.getChildren().add(new Label("Condition: " + condition));
        resultBox.getChildren().add(new Label("Recommended Course of Action: " + recommendedCourse));

        resultPane.setCenter(resultBox);

        Scene resultScene = new Scene(resultPane, 400, 250);
        resultStage.setScene(resultScene);
        resultStage.setTitle("Results");
        resultStage.show();
    }
    
    private void varTGIthree (float stddev, float eightystddev) {
    	double factor = stddev/eightystddev; 
    	tgiOut =(float) (10 * Math.pow(0.675, factor));
    	
        resultStage = new Stage();
        BorderPane resultPane = new BorderPane();
        resultPane.setPadding(new Insets(10));

        VBox resultBox = new VBox(10);
        resultBox.setPadding(new Insets(10));
        resultBox.setAlignment(Pos.CENTER_LEFT);

        resultBox.getChildren().add(new Label("TGI Output: " + tgiOut));

        resultPane.setCenter(resultBox);

        Scene resultScene = new Scene(resultPane, 400, 250);
        resultStage.setScene(resultScene);
        resultStage.setTitle("Results");
        resultStage.show();
    }
    
    private void varTGIfour (float facOne, float facTwo, float facThree, float facFour, float facFive, float facSix) {
    	float factor = facOne/facTwo; 
    	tgiOut = 100 * factor;
    	
    	float factor2 = facThree/facFour;
    	float tgiOut2 = 100 * factor2;
    	
    	float factor3 = facFive/facSix;
    	float tgiOut3 = 100 * factor3;
    	
        resultStage = new Stage();
        BorderPane resultPane = new BorderPane();
        resultPane.setPadding(new Insets(10));

        VBox resultBox = new VBox(10);
        resultBox.setPadding(new Insets(10));
        resultBox.setAlignment(Pos.CENTER_LEFT);

        resultBox.getChildren().add(new Label("K Output (longitudinal) : " + tgiOut));
        resultBox.getChildren().add(new Label("K Output (alignment) : " + tgiOut2));
        resultBox.getChildren().add(new Label("K Output (gauge) : " + tgiOut3));

        resultPane.setCenter(resultBox);

        Scene resultScene = new Scene(resultPane, 400, 250);
        resultStage.setScene(resultScene);
        resultStage.setTitle("Results");
        resultStage.show();
    }
    
    private void varTGIfive (float H, float Hlim, float S, float Slim) {
    	float normalizedH = H / Hlim;
        float normalizedS = 2 * (S / Slim);
        
        
        float totalDeviation = normalizedH + normalizedS;
        
        
        int roundedDeviation = (int) Math.ceil(totalDeviation);
        
        
        tgiOut = 150 - (100.0f / 3.0f) * roundedDeviation;
    	
        resultStage = new Stage();
        BorderPane resultPane = new BorderPane();
        resultPane.setPadding(new Insets(10));

        VBox resultBox = new VBox(10);
        resultBox.setPadding(new Insets(10));
        resultBox.setAlignment(Pos.CENTER_LEFT);

        resultBox.getChildren().add(new Label("QI Output: " + tgiOut));

        resultPane.setCenter(resultBox);

        Scene resultScene = new Scene(resultPane, 400, 250);
        resultStage.setScene(resultScene);
        resultStage.setTitle("Results");
        resultStage.show();
    }
}