import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.RadioButton;
import javafx.scene.control.TextField;
import javafx.scene.control.ToggleGroup;
import javafx.scene.input.KeyCode;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
import javafx.stage.FileChooser;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;


public class RailGeometry {
	
	// helper methods and globals 

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
    
    public static float calc80(float[] values) { //helper method for 80th percentile 
        if (values == null || values.length == 0) {
            throw new IllegalArgumentException("Array must contain at least one element.");
        }
        
        Arrays.sort(values);

        int index = (int) Math.ceil(0.8 * values.length) - 1; 

        return values[index];
    }
    

    private void showError(String message) { //error message
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle("Input Error");
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }

    private void configureBoundedInputField(TextField textField) { //bounded field 
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
    
    private void configureInputField(TextField textField) { // input text
        textField.setPromptText("Enter a positive number");
    }

    private void configurePositiveInputField(TextField textField) { // replicate
        textField.setPromptText("Enter a positive number");
    }
    
    
    private float[] parseInputToFloatArray(String input) {
        // helper function, parse formatted input to array
        String[] stringValues = input.trim().split("\\s*,\\s*");
        float[] floatValues = new float[stringValues.length];

        for (int i = 0; i < stringValues.length; i++) {
            floatValues[i] = Float.parseFloat(stringValues[i]);
        }

        return floatValues;
    }
    
 
    private void collectInstanceDataDefault(List<float[]> longitudinalList, List<float[]> alignmentList, List<float[]> gaugeList, int instanceNumber) {
    	// Helper function to collect input data for each instance, default
        Stage instanceWindow = new Stage();
        BorderPane pane = new BorderPane();
        pane.setPadding(new Insets(10));

        GridPane gridPane = new GridPane();
        gridPane.setPadding(new Insets(10));
        gridPane.setHgap(10);
        gridPane.setVgap(10);
        gridPane.setAlignment(Pos.CENTER_LEFT);

        Label instanceLabel = new Label("Instance " + instanceNumber);
        instanceLabel.setStyle("-fx-font-weight: bold;");

        // Input fields for longitudinal, alignment, and gauge deviation
        Label longitudinalLabel = new Label("Longitudinal Deviation (L): ");
        TextField longitudinalInput = new TextField();
        Label alignmentLabel = new Label("Alignment Deviation (A): ");
        TextField alignmentInput = new TextField();
        Label gaugeLabel = new Label("Gauge Deviation (G): ");
        TextField gaugeInput = new TextField();

        Button submitButton = new Button("Submit");
        submitButton.setOnAction(e -> {
            try {
                longitudinalList.add(parseInputToFloatArray(longitudinalInput.getText()));
                alignmentList.add(parseInputToFloatArray(alignmentInput.getText()));
                gaugeList.add(parseInputToFloatArray(gaugeInput.getText()));
                instanceWindow.close();
            } catch (NumberFormatException ex) {
                showError("Please enter valid numerical values.");
            }
        });

        gridPane.add(instanceLabel, 0, 0);
        gridPane.add(longitudinalLabel, 0, 1);
        gridPane.add(longitudinalInput, 1, 1);
        gridPane.add(alignmentLabel, 0, 2);
        gridPane.add(alignmentInput, 1, 2);
        gridPane.add(gaugeLabel, 0, 3);
        gridPane.add(gaugeInput, 1, 3);
        gridPane.add(submitButton, 1, 4);

        pane.setCenter(gridPane);

        Scene scene = new Scene(pane, 400, 300);
        instanceWindow.setScene(scene);
        instanceWindow.setTitle("Instance " + instanceNumber + " Input");
        instanceWindow.showAndWait();
    }

    private void collectInstanceData(List<float[]> HLeftList, List<float[]> HRightList, List<float[]> crossLevelsList, List<float[]> gaugesList, List<float[]> horizontalDeviationsList, int instanceNumber) {
    	//helper function, fetch list for instances, swedenQ
        Stage instanceWindow = new Stage();
        BorderPane pane = new BorderPane();
        pane.setPadding(new Insets(10));

        GridPane gridPane = new GridPane();
        gridPane.setPadding(new Insets(10));
        gridPane.setHgap(10);
        gridPane.setVgap(10);
        gridPane.setAlignment(Pos.CENTER_LEFT);

        Label instanceLabel = new Label("Instance " + instanceNumber);
        instanceLabel.setStyle("-fx-font-weight: bold;");

        Label hLeftLabel = new Label("H Left (Comma Separated List): ");
        TextField hLeftInput = new TextField();

        Label hRightLabel = new Label("H Right (Comma Separated List): ");
        TextField hRightInput = new TextField();

        Label crossLevelsLabel = new Label("Cross Levels (Comma Separated List): ");
        TextField crossLevelsInput = new TextField();

        Label gaugesLabel = new Label("Gauges (Comma Separated List): ");
        TextField gaugesInput = new TextField();

        Label horizontalDeviationsLabel = new Label("Horizontal Deviations (Comma Separated List): ");
        TextField horizontalDeviationsInput = new TextField();

        Button submitButton = new Button("Submit");

        submitButton.setOnAction(e -> {
            try {
                HLeftList.add(parseInputToFloatArray(hLeftInput.getText()));
                HRightList.add(parseInputToFloatArray(hRightInput.getText()));
                crossLevelsList.add(parseInputToFloatArray(crossLevelsInput.getText()));
                gaugesList.add(parseInputToFloatArray(gaugesInput.getText()));
                horizontalDeviationsList.add(parseInputToFloatArray(horizontalDeviationsInput.getText()));
                instanceWindow.close();
            } catch (NumberFormatException ex) {
                showError("Please enter valid numerical values.");
            }
        });

        gridPane.add(instanceLabel, 0, 0);
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
        gridPane.add(submitButton, 1, 6);

        pane.setCenter(gridPane);

        Scene scene = new Scene(pane, 400, 300);
        instanceWindow.setScene(scene);
        instanceWindow.setTitle("Instance " + instanceNumber + " Input");
        instanceWindow.showAndWait();
    }
    
    private float[] getTrackClassLimits(int trackClass) {
    	//helper function, defines track class limits
        switch (trackClass) {
            case 1: return new float[] {3, 3};
            case 2: return new float[] {2, 2};
            case 3: return new float[] {1.75f, 1.75f};
            case 4: return new float[] {1.5f, 1.5f};
            case 5: return new float[] {1, 1};
            default: throw new IllegalArgumentException("Invalid Track Class");
        }
    }
    
    // main driver method
    
    public void openMenuSelection(Stage primaryStage) {
        BorderPane root = new BorderPane();
        root.setPadding(new Insets(10));

        VBox optionsBox = new VBox(10);
        optionsBox.setPadding(new Insets(10));
        optionsBox.setAlignment(Pos.CENTER_LEFT);

        Label instructionLabel = new Label("Please select an option:");
        optionsBox.getChildren().add(instructionLabel);

        ToggleGroup toggleGroup = new ToggleGroup();

        for (int i = 1; i <= 9; i++) {
            String optionName;
            
            switch (i) {
                case 1:
                    optionName = "Default";
                    break;
                case 2:
                    optionName = "Variation 1";
                    break;
                case 3:
                    optionName = "Variation 2";
                    break;
                case 4:
                    optionName = "Netherlands Track Quality Index";
                    break;
                case 5:
                    optionName = "Sweden Q";
                    break;
                case 6:
                    optionName = "J Coefficient";
                    break;
                case 7:
                    optionName = "CN Index";
                    break;
                case 8:
                    optionName = "Track Geometry Index";
                    break;
                case 9:
                    optionName = "Track Geometry Index Variation";
                    break;
                default:
                    optionName = "Default";
            }

            final String selectedOption = optionName;
            RadioButton variantOption = new RadioButton(optionName);
            variantOption.setToggleGroup(toggleGroup);
            variantOption.setOnAction(e -> userChoice = selectedOption);
            optionsBox.getChildren().add(variantOption);
        }

        root.setLeft(optionsBox);

        Button enterButton = new Button("Enter");
        enterButton.setOnAction(e -> {
            System.out.println("User Choice: " + userChoice);

            switch (userChoice) {
                case "Default":
                    openDefaultWindow();
                    break;
                case "Variation 1":
                    openVarOneWindow(); 
                    break;
                case "Variation 2":
                    openVarTwoWindow(); 
                    break;
                case "Netherlands Track Quality Index":
                    openNTQIWindow();
                    break;
                case "Sweden Q":
                    openSwedenQWindow();
                    break;
                case "J Coefficient":
                    openJCoeffWindow();
                    break;
                case "CN Index":
                    openCNWindow();
                    break;
                case "Track Geometry Index":
                    openTGIWindow();
                    break;
                case "Track Geometry Index Variation":
                    openTGIVarWindow();
                    break;
                default:
                    primaryStage.close();
                    break;
            }
        });

        BorderPane bottomPane = new BorderPane();
        bottomPane.setRight(enterButton);
        BorderPane.setMargin(enterButton, new Insets(10));
        root.setBottom(bottomPane);

        Scene scene = new Scene(root, 350, 350);
        primaryStage.setScene(scene);
        primaryStage.setTitle("Menu Selection");
    }
    
    // input methods, mostly similar in construction

    private void openDefaultWindow() {
        Stage defaultWindow = new Stage();
        BorderPane pane = new BorderPane();
        pane.setPadding(new Insets(10));

        GridPane gridPane = new GridPane();
        gridPane.setPadding(new Insets(10));
        gridPane.setHgap(10);
        gridPane.setVgap(10);
        gridPane.setAlignment(Pos.CENTER_LEFT);

        Label notice = new Label("All measurements in inches");
        notice.setStyle("-fx-font-weight: bold;");

        // Input fields for the number of instances, track class, and track type
        Label instancesLabel = new Label("Number of Instances: ");
        TextField instancesInput = new TextField();
        configureInputField(instancesInput);

        Label classLabel = new Label("Class of Track: ");
        ComboBox<String> classCombo = new ComboBox<>();
        classCombo.getItems().addAll("1", "2", "3", "4", "5");

        Label typeLabel = new Label("Type of Track: ");
        ComboBox<String> typeCombo = new ComboBox<>();
        typeCombo.getItems().addAll("Line (Straight)", "31-foot Chord", "62-foot Chord", "31-foot Qualified Cant Chord", "62-foot Qualified Cant Chord");

        // Add components to grid
        gridPane.add(notice, 0, 0);
        gridPane.add(instancesLabel, 0, 1);
        gridPane.add(instancesInput, 1, 1);
        gridPane.add(classLabel, 0, 2);
        gridPane.add(classCombo, 1, 2);
        gridPane.add(typeLabel, 0, 3);
        gridPane.add(typeCombo, 1, 3);

        pane.setCenter(gridPane);

        Button enterButton = new Button("Enter");
        enterButton.setOnAction(e -> {
            try {
                int instances = Integer.parseInt(instancesInput.getText());

                // Get selected class and type
                int trackClass = Integer.parseInt(classCombo.getValue());
                String trackType = typeCombo.getValue();

                // Variables for allowances (set based on track class and type)
                float Lmax = 0, Amax = 0, Gmax = 0;

                if (trackType.equals("Line (Straight)")) {
                    switch (trackClass) {
                        case 1 -> { Lmax = 3; Gmax = 1; Amax = 5; }
                        case 2 -> { Lmax = 2; Gmax = 0.875f; Amax = 3; }
                        case 3 -> { Lmax = 1.75f; Gmax = 0.875f; Amax = 1.75f; }
                        case 4 -> { Lmax = 1.5f; Gmax = 0.75f; Amax = 1.5f; }
                        case 5 -> { Lmax = 1; Gmax = 0.75f; Amax = 0.75f; }
                    }
                } else if (trackType.equals("31-foot Chord")) {
                    switch (trackClass) {
                        case 1 -> { Lmax = 0; Gmax = 1; Amax = 0; }
                        case 2 -> { Lmax = 0; Gmax = 0.875f; Amax = 0; }
                        case 3 -> { Lmax = 1; Gmax = 0.875f; Amax = 1.25f; }
                        case 4 -> { Lmax = 1; Gmax = 0.75f; Amax = 1; }
                        case 5 -> { Lmax = 1; Gmax = 0.75f; Amax = 0.5f; }
                    }
                } else if (trackType.equals("62-foot Chord")) {
                    switch (trackClass) {
                        case 1 -> { Lmax = 3; Gmax = 1; Amax = 5; }
                        case 2 -> { Lmax = 2.75f; Gmax = 0.875f; Amax = 3; }
                        case 3 -> { Lmax = 2.25f; Gmax = 0.875f; Amax = 1.75f; }
                        case 4 -> { Lmax = 2; Gmax = 0.75f; Amax = 1.5f; }
                        case 5 -> { Lmax = 1.25f; Gmax = 0.75f; Amax = 0.625f; }
                    }
                } else if (trackType.equals("31-foot Qualified Cant Chord")) {
                    switch (trackClass) {
                        case 1 -> { Lmax = 0; Gmax = 1; Amax = 0; }
                        case 2 -> { Lmax = 0; Gmax = 0.875f; Amax = 0; }
                        case 3 -> { Lmax = 1; Gmax = 0.875f; Amax = 1.25f; }
                        case 4 -> { Lmax = 1; Gmax = 0.75f; Amax = 1; }
                        case 5 -> { Lmax = 1; Gmax = 0.75f; Amax = 0.5f; }
                    }
                } else if (trackType.equals("62-foot Qualified Cant Chord")) {
                    switch (trackClass) {
                        case 1 -> { Lmax = 2.25f; Gmax = 1; Amax = 1.25f; }
                        case 2 -> { Lmax = 2.25f; Gmax = 0.875f; Amax = 1.25f; }
                        case 3 -> { Lmax = 1.75f; Gmax = 0.875f; Amax = 1.25f; }
                        case 4 -> { Lmax = 1.25f; Gmax = 0.75f; Amax = 0.875f; }
                        case 5 -> { Lmax = 1; Gmax = 0.75f; Amax = 0.625f; }
                    }
                }

                // Now that we have track class/type and limits, collect data for multiple instances
                List<float[]> longitudinalList = new ArrayList<>();
                List<float[]> alignmentList = new ArrayList<>();
                List<float[]> gaugeList = new ArrayList<>();

                for (int i = 0; i < instances; i++) {
                    collectInstanceDataDefault(longitudinalList, alignmentList, gaugeList, i + 1);
                }

                // After collecting data, perform calculations
                varDefaultTGI(instances, Lmax, Amax, Gmax, longitudinalList, alignmentList, gaugeList);
                defaultWindow.close();
            } catch (Exception ex) {
                showError("Invalid input. Please enter valid numbers.");
            }
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
    
    private void openNTQIWindow() {
        Stage NTQIWindow = new Stage();
        BorderPane pane = new BorderPane();
        pane.setPadding(new Insets(10));

        GridPane gridPane = new GridPane();
        gridPane.setPadding(new Insets(10));
        gridPane.setHgap(10);
        gridPane.setVgap(10);
        gridPane.setAlignment(Pos.CENTER_LEFT);

        Label notice = new Label("All measurements in mm");
        notice.setStyle("-fx-font-weight: bold;");

        Label rangeLabel = new Label("Enter Range of Values (Comma Separated): ");
        TextField rangeInput = new TextField();
        configureInputField(rangeInput);

        gridPane.add(notice, 0, 0);
        gridPane.add(rangeLabel, 0, 1);
        gridPane.add(rangeInput, 1, 1);
        
        pane.setCenter(gridPane);

        Button enterButton = new Button("Enter");
        enterButton.setOnAction(e -> {
            try {
                float[] values = parseInputToFloatArray(rangeInput.getText());

                double stdDev = stdv(values);
                float percentile80 = calc80(values);

                varTGIntqi((float) stdDev, percentile80);
                NTQIWindow.close();
            } catch (NumberFormatException ex) {
                showError("Please enter valid numerical values.");
            } catch (IllegalArgumentException ex) {
                showError(ex.getMessage());
            }
        });

        BorderPane bottomPane = new BorderPane();
        bottomPane.setRight(enterButton);
        BorderPane.setMargin(enterButton, new Insets(10));
        pane.setBottom(bottomPane);

        Scene scene = new Scene(pane, 400, 300);
        NTQIWindow.setScene(scene);
        NTQIWindow.setTitle("NTQI Input");
        NTQIWindow.show();
    }
    
    private void openSwedenQWindow() {
        Stage swedenQWindow = new Stage();
        BorderPane pane = new BorderPane();
        pane.setPadding(new Insets(10));

        GridPane gridPane = new GridPane();
        gridPane.setPadding(new Insets(10));
        gridPane.setHgap(10);
        gridPane.setVgap(10);
        gridPane.setAlignment(Pos.CENTER_LEFT);

        Label notice = new Label("All measurements in mm");
        notice.setStyle("-fx-font-weight: bold;");

        // Number of Instances
        Label instanceLabel = new Label("Number of Instances: ");
        TextField instanceInput = new TextField();
        configurePositiveInputField(instanceInput);

        // Track Length
        Label lengthLabel = new Label("Track Length (ft): ");
        TextField lengthInput = new TextField();
        configurePositiveInputField(lengthInput);

        // Track Class
        Label classLabel = new Label("Track Class: ");
        ComboBox<String> classCombo = new ComboBox<>();
        classCombo.getItems().addAll("1", "2", "3", "4", "5");

        Button nextButton = new Button("Next");

        // Go to instance input when the number of instances is provided
        nextButton.setOnAction(e -> {
            try {
                int instances = Integer.parseInt(instanceInput.getText());
                float trackLength = Float.parseFloat(lengthInput.getText());
                int trackClass = Integer.parseInt(classCombo.getValue());

                if (instances <= 0 || trackLength <= 0) {
                    throw new NumberFormatException("Values must be positive.");
                }

                float[] limits = getTrackClassLimits(trackClass); // Get Hlim and Slim based on class

                List<float[]> HLeftList = new ArrayList<>();
                List<float[]> HRightList = new ArrayList<>();
                List<float[]> crossLevelsList = new ArrayList<>();
                List<float[]> gaugesList = new ArrayList<>();
                List<float[]> horizontalDeviationsList = new ArrayList<>();

                // Loop through each instance and gather input
                for (int i = 0; i < instances; i++) {
                    collectInstanceData(HLeftList, HRightList, crossLevelsList, gaugesList, horizontalDeviationsList, i + 1);
                }

                // Pass the data to varTGIfive after all inputs are collected
                varTGIswedenQ(instances, trackLength, limits[0], limits[1], HLeftList, HRightList, crossLevelsList, gaugesList, horizontalDeviationsList);
                swedenQWindow.close();

            } catch (NumberFormatException ex) {
                showError("Please enter valid positive numerical values.");
            }
        });

        gridPane.add(notice, 0, 0);
        gridPane.add(instanceLabel, 0, 1);
        gridPane.add(instanceInput, 1, 1);
        gridPane.add(lengthLabel, 0, 2);
        gridPane.add(lengthInput, 1, 2);
        gridPane.add(classLabel, 0, 3);
        gridPane.add(classCombo, 1, 3);
        gridPane.add(nextButton, 1, 4);

        pane.setCenter(gridPane);

        Scene scene = new Scene(pane, 400, 300);
        swedenQWindow.setScene(scene);
        swedenQWindow.setTitle("Sweden Q Input");
        swedenQWindow.show();
    }
    
    private void openJCoeffWindow() {
        Stage jCoeffWindow = new Stage();
        BorderPane pane = new BorderPane();
        pane.setPadding(new Insets(10));

        GridPane gridPane = new GridPane();
        gridPane.setPadding(new Insets(10));
        gridPane.setHgap(10);
        gridPane.setVgap(10);
        gridPane.setAlignment(Pos.CENTER_LEFT);

        Label notice = new Label("All measurements in mm");
        notice.setStyle("-fx-font-weight: bold;");

        Label ZLabel = new Label("Longitudinal Level Values (Comma Separated list): ");
        TextField ZInput = new TextField();
        
        Label YLabel = new Label("Alignment Values(Comma Separated list): ");
        TextField YInput = new TextField();

        Label WLabel = new Label("Twist values (Comma Separated list): ");
        TextField WInput = new TextField();

        Label ELabel = new Label("Track Gauge values (Comma Separated list): ");
        TextField EInput = new TextField();

        gridPane.add(notice, 0, 0);
        gridPane.add(ZLabel, 0, 1);
        gridPane.add(ZInput, 1, 1);
        gridPane.add(YLabel, 0, 2);
        gridPane.add(YInput, 1, 2);
        gridPane.add(WLabel, 0, 3);
        gridPane.add(WInput, 1, 3);
        gridPane.add(ELabel, 0, 4);
        gridPane.add(EInput, 1, 4);
       
        pane.setCenter(gridPane);

        Button enterButton = new Button("Enter");
        enterButton.setOnAction(e -> {
            try {
                float[] Z = parseInputToFloatArray(ZInput.getText());
                float[] Y = parseInputToFloatArray(YInput.getText());
                float[] W = parseInputToFloatArray(WInput.getText());
                float[] E = parseInputToFloatArray(EInput.getText());
                
                float SDz = (float)stdv(Z);
                float SDy = (float)stdv(Y);
                float SDw = (float)stdv(W);
                float SDe = (float)stdv(E);
                varTGIjCoeff(SDz, SDy, SDw, SDe);
                jCoeffWindow.close();
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
        jCoeffWindow.setScene(scene);
        jCoeffWindow.setTitle("JCoeff Input");
        jCoeffWindow.show();
    }
    
    private void openCNWindow() { // this allows user input for testing. The list values need to be read in via excel.
        Stage CNWindow = new Stage();
        BorderPane pane = new BorderPane();
        pane.setPadding(new Insets(10));

        GridPane gridPane = new GridPane();
        gridPane.setPadding(new Insets(10));
        gridPane.setHgap(10);
        gridPane.setVgap(10);
        gridPane.setAlignment(Pos.CENTER_LEFT);

        Label notice = new Label("All measurements in mm");
        notice.setStyle("-fx-font-weight: bold;");

        Label gaugeLabel = new Label("Gauge (Comma Separated list): ");
        TextField gaugeInput = new TextField();
        
        Label crossLevelLabel = new Label("Cross Level (Comma Separated list): ");
        TextField crossLevelInput = new TextField();

        Label leftSurfaceLabel = new Label("Left Surface (Comma Separated list): ");
        TextField leftSurfaceInput = new TextField();

        Label rightSurfaceLabel = new Label("Right Surface (Comma Separated list): ");
        TextField rightSurfaceInput = new TextField();

        Label leftAlignmentLabel = new Label("Left Alignment (Comma Separated list): ");
        TextField leftAlignmentInput = new TextField();

        Label rightAlignmentLabel = new Label("Right Alignment (Comma Separated list): ");
        TextField rightAlignmentInput = new TextField();
        
        gridPane.add(notice, 0, 0);
        gridPane.add(gaugeLabel, 0, 1);
        gridPane.add(gaugeInput, 1, 1);
        gridPane.add(crossLevelLabel, 0, 2);
        gridPane.add(crossLevelInput, 1, 2);
        gridPane.add(leftSurfaceLabel, 0, 3);
        gridPane.add(leftSurfaceInput, 1, 3);
        gridPane.add(rightSurfaceLabel, 0, 4);
        gridPane.add(rightSurfaceInput, 1, 4);
        gridPane.add(leftAlignmentLabel, 0, 5);
        gridPane.add(leftAlignmentInput, 1, 5);
        gridPane.add(rightAlignmentLabel, 0, 6);
        gridPane.add(rightAlignmentInput, 1, 6);

        pane.setCenter(gridPane);

        Button enterButton = new Button("Enter");
        enterButton.setOnAction(e -> {
            try {
                float[] gauges = parseInputToFloatArray(gaugeInput.getText());
                float[] crossLevels = parseInputToFloatArray(crossLevelInput.getText());
                float[] leftSurfaces = parseInputToFloatArray(leftSurfaceInput.getText());
                float[] rightSurfaces = parseInputToFloatArray(rightSurfaceInput.getText());
                float[] leftAlignments = parseInputToFloatArray(leftAlignmentInput.getText());
                float[] rightAlignments = parseInputToFloatArray(rightAlignmentInput.getText());
                
                float stdvGauge = (float)stdv(gauges);
                float stdvCross = (float)stdv(crossLevels);
                float stdvLeftS = (float)stdv(leftSurfaces);
                float stdvRightS = (float)stdv(rightSurfaces);
                float stdvLeftA = (float)stdv(leftAlignments);
                float stdvRightA = (float)stdv(rightAlignments);

                varTGIcn(stdvGauge, stdvCross, stdvLeftA, stdvRightA, stdvLeftS, stdvRightS);
                CNWindow.close();
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
        CNWindow.setScene(scene);
        CNWindow.setTitle("CN Input");
        CNWindow.show();
    }
    
    private void openTGIWindow() {
        Stage TGIWindow = new Stage();
        BorderPane pane = new BorderPane();
        pane.setPadding(new Insets(10));

        GridPane gridPane = new GridPane();
        gridPane.setPadding(new Insets(10));
        gridPane.setHgap(10);
        gridPane.setVgap(10);
        gridPane.setAlignment(Pos.CENTER_LEFT);

        Label notice = new Label("All measurements in mm");
        notice.setStyle("-fx-font-weight: bold;");

        // Radio buttons for speed selection
        ToggleGroup speedGroup = new ToggleGroup();
        RadioButton above105 = new RadioButton("> 105 kph");
        above105.setToggleGroup(speedGroup);
        RadioButton below105 = new RadioButton("< 105 kph");
        below105.setToggleGroup(speedGroup);
        
        Label hLeftLabel = new Label("Uneveness Level (Comma Separated list): ");
        TextField hLeftInput = new TextField();

        Label hRightLabel = new Label("Alignment (Comma Separated list): ");
        TextField hRightInput = new TextField();

        Label crossLevelsLabel = new Label("Gauge (Comma Separated list): ");
        TextField crossLevelsInput = new TextField();

        Label gaugesLabel = new Label("Twist (Comma Separated list): ");
        TextField gaugesInput = new TextField();

        gridPane.add(notice, 0, 0);
        gridPane.add(above105, 0, 1);
        gridPane.add(below105, 0, 2);
        gridPane.add(hLeftLabel, 0, 3);
        gridPane.add(hLeftInput, 1, 3);
        gridPane.add(hRightLabel, 0, 4);
        gridPane.add(hRightInput, 1, 4);
        gridPane.add(crossLevelsLabel, 0, 5);
        gridPane.add(crossLevelsInput, 1, 5);
        gridPane.add(gaugesLabel, 0, 6);
        gridPane.add(gaugesInput, 1, 6);

        pane.setCenter(gridPane);

        Button enterButton = new Button("Enter");
        enterButton.setOnAction(e -> {
            try {
                float SDnewLong = 2.5f;
                float SDnewAlign = 1.5f;
                float SDnewGauge = 1.0f;
                float SDnewTwist = 1.75f;

                float SDmainLong = above105.isSelected() ? 6.2f : 7.2f;
                float SDmainAlign = 3.0f;
                float SDmainGauge = 3.6f;
                float SDmainTwist = above105.isSelected() ? 3.8f : 4.2f;

                float[] longitudinalLevel = parseInputToFloatArray(hLeftInput.getText());
                float[] alignment = parseInputToFloatArray(hRightInput.getText());
                float[] gauge = parseInputToFloatArray(crossLevelsInput.getText());
                float[] twist = parseInputToFloatArray(gaugesInput.getText());
                
                float SDu = (float)stdv(longitudinalLevel);
                float SDa = (float)stdv(alignment);
                float SDg = (float)stdv(gauge);
                float SDt = (float)stdv(twist);

                varTGI(SDu, SDa, SDt, SDg, SDnewLong, SDnewAlign, SDnewTwist, SDnewGauge, SDmainLong, SDmainAlign, SDmainTwist,SDmainGauge);
                TGIWindow.close();
            } catch (NumberFormatException ex) {
                showError("Please enter valid numerical values.");
            } catch (IllegalArgumentException ex) {
                showError(ex.getMessage());
            }
        });

        BorderPane bottomPane = new BorderPane();
        bottomPane.setRight(enterButton);
        BorderPane.setMargin(enterButton, new Insets(10));
        pane.setBottom(bottomPane);

        Scene scene = new Scene(pane, 400, 400);
        TGIWindow.setScene(scene);
        TGIWindow.setTitle("TGI Input");
        TGIWindow.show();
    }
    
    private void openTGIVarWindow() {
        Stage TGIVarWindow = new Stage();
        BorderPane pane = new BorderPane();
        pane.setPadding(new Insets(10));

        GridPane gridPane = new GridPane();
        gridPane.setPadding(new Insets(10));
        gridPane.setHgap(10);
        gridPane.setVgap(10);
        gridPane.setAlignment(Pos.CENTER_LEFT);

        Label notice = new Label("All measurements in inches");
        notice.setStyle("-fx-font-weight: bold;");

        Label classLabel = new Label("Class of Track: ");
        ComboBox<String> classCombo = new ComboBox<>();
        classCombo.getItems().addAll("1", "2", "3", "4", "5");

        Label typeLabel = new Label("Type of Track: ");
        ComboBox<String> typeCombo = new ComboBox<>();
        typeCombo.getItems().addAll("Line (Straight)", "31-foot Chord", "62-foot Chord", "31-foot Qualified Cant Chord", "62-foot Qualified Cant Chord");

        Label longitudinalLabel = new Label("Longitudinal Deviation (L): ");
        TextField longitudinalInput = new TextField();
        configureInputField(longitudinalInput);

        Label alignmentLabel = new Label("Alignment Deviation (A): ");
        TextField alignmentInput = new TextField();
        configureInputField(alignmentInput);

        Label gaugeLabel = new Label("Gauge Deviation (G): ");
        TextField gaugeInput = new TextField();
        configureInputField(gaugeInput);
        
        gridPane.add(notice, 0, 0);
        gridPane.add(classLabel, 0, 1);
        gridPane.add(classCombo, 1, 1);
        gridPane.add(typeLabel, 0, 2);
        gridPane.add(typeCombo, 1, 2);
        gridPane.add(longitudinalLabel, 0, 3);
        gridPane.add(longitudinalInput, 1, 3);
        gridPane.add(alignmentLabel, 0, 4);
        gridPane.add(alignmentInput, 1, 4);
        gridPane.add(gaugeLabel, 0, 5);
        gridPane.add(gaugeInput, 1, 5);

        pane.setCenter(gridPane);

        Button enterButton = new Button("Enter");
        enterButton.setOnAction(e -> {
            int trackClass = Integer.parseInt(classCombo.getValue());
            String trackType = typeCombo.getValue();
            float l = Float.parseFloat(longitudinalInput.getText());
            float a = Float.parseFloat(alignmentInput.getText());
            float g = Float.parseFloat(gaugeInput.getText());

            float Lmax = 0, Gmax = 0, Amax = 0;

            if (trackType.equals("Line (Straight)")) {
                switch (trackClass) {
                    case 1 -> { Lmax = 3; Gmax = 1; Amax = 5; }
                    case 2 -> { Lmax = 2; Gmax = 0.875f; Amax = 3; }
                    case 3 -> { Lmax = 1.75f; Gmax = 0.875f; Amax = 1.75f; }
                    case 4 -> { Lmax = 1.5f; Gmax = 0.75f; Amax = 1.5f; }
                    case 5 -> { Lmax = 1; Gmax = 0.75f; Amax = 0.75f; }
                }
            }
            else if (trackType.equals("31-foot Chord")) {
                switch (trackClass) {
                    case 1 -> { Lmax = 0; Gmax = 1; Amax = 0; }  // N/A for L and A
                    case 2 -> { Lmax = 0; Gmax = 0.875f; Amax = 0; }  // N/A for L and A
                    case 3 -> { Lmax = 1; Gmax = 0.875f; Amax = 1.25f; }
                    case 4 -> { Lmax = 1; Gmax = 0.75f; Amax = 1; }
                    case 5 -> { Lmax = 1; Gmax = 0.75f; Amax = 0.5f; }
                }
            }
            else if (trackType.equals("62-foot Chord")) {
                switch (trackClass) {
                    case 1 -> { Lmax = 3; Gmax = 1; Amax = 5; }
                    case 2 -> { Lmax = 2.75f; Gmax = 0.875f; Amax = 3; }
                    case 3 -> { Lmax = 2.25f; Gmax = 0.875f; Amax = 1.75f; }
                    case 4 -> { Lmax = 2; Gmax = 0.75f; Amax = 1.5f; }
                    case 5 -> { Lmax = 1.25f; Gmax = 0.75f; Amax = 0.625f; }
                }
            }
            else if (trackType.equals("31-foot Qualified Cant Chord")) {
                switch (trackClass) {
                    case 1 -> { Lmax = 0; Gmax = 1; Amax = 0; }  // N/A for L and A
                    case 2 -> { Lmax = 0; Gmax = 0.875f; Amax = 0; }  // N/A for L and A
                    case 3 -> { Lmax = 1; Gmax = 0.875f; Amax = 1.25f; }
                    case 4 -> { Lmax = 1; Gmax = 0.75f; Amax = 1; }
                    case 5 -> { Lmax = 1; Gmax = 0.75f; Amax = 0.5f; }
                }
            }
            else if (trackType.equals("62-foot Qualified Cant Chord")) {
                switch (trackClass) {
                    case 1 -> { Lmax = 2.25f; Gmax = 1; Amax = 1.25f; }
                    case 2 -> { Lmax = 2.25f; Gmax = 0.875f; Amax = 1.25f; }
                    case 3 -> { Lmax = 1.75f; Gmax = 0.875f; Amax = 1.25f; }
                    case 4 -> { Lmax = 1.25f; Gmax = 0.75f; Amax = 0.875f; }
                    case 5 -> { Lmax = 1; Gmax = 0.75f; Amax = 0.625f; }
                }
            }

            varTGIvar(l, a, g, Lmax, Amax, Gmax);

            TGIVarWindow.close();
        });

        BorderPane bottomPane = new BorderPane();
        bottomPane.setRight(enterButton);
        BorderPane.setMargin(enterButton, new Insets(10));
        pane.setBottom(bottomPane);

        Scene scene = new Scene(pane, 400, 300);
        TGIVarWindow.setScene(scene);
        TGIVarWindow.setTitle("TGI Variation Input");
        TGIVarWindow.show();
    }
     
    // output methods, assumes that we have the components necessary for final computation, displays output
    
    private void varDefaultTGI(int instances, float Lmax, float Amax, float Gmax, List<float[]> longitudinalList, List<float[]> alignmentList, List<float[]> gaugeList) {
        List<Float> tgiValues = new ArrayList<>();
        int satisfactoryInstances = 0;

        StringBuilder exceedanceOutput = new StringBuilder();

        for (int i = 0; i < instances; i++) {
            float l = longitudinalList.get(i)[0];
            float a = alignmentList.get(i)[0];
            float g = gaugeList.get(i)[0];

            float exceedL = l - Lmax > 0 ? l - Lmax : 0;
            float exceedA = a - Amax > 0 ? a - Amax : 0;
            float exceedG = g - Gmax > 0 ? g - Gmax : 0;

            float tgiOut = 100 - ((l + a + g) / (Lmax + Amax + Gmax)) * 100;
            tgiOut = Math.round(tgiOut);

            tgiValues.add(tgiOut);

            if (tgiOut >= 0) {
                satisfactoryInstances++;
            }

            exceedanceOutput.append(String.format("Instance %d L, A, G exceed: %.2f, %.2f, %.2f\n", i + 1, exceedL, exceedA, exceedG));
        }

        float K = (float) satisfactoryInstances / instances;

        resultStage = new Stage();
        BorderPane resultPane = new BorderPane();
        resultPane.setPadding(new Insets(10));

        VBox resultBox = new VBox(10);
        resultBox.setPadding(new Insets(10));
        resultBox.setAlignment(Pos.CENTER_LEFT);

        resultBox.getChildren().add(new Label("TGI Values: " + tgiValues.toString()));
        resultBox.getChildren().add(new Label("K Value: " + String.format("%.2f", K)));

        resultBox.getChildren().add(new Label("Exceedance Values:"));
        resultBox.getChildren().add(new Label(exceedanceOutput.toString()));

        resultPane.setCenter(resultBox);

        Scene resultScene = new Scene(resultPane, 400, 400);
        resultStage.setScene(resultScene);
        resultStage.setTitle("Default TGI Results");
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
    
    private void varTGIntqi (float stddev, float eightystddev) {
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
   
    private void varTGIswedenQ(int instances, float trackLength, float Hlim, float Slim, List<float[]> HLeftList, List<float[]> HRightList, List<float[]> crossLevelsList, List<float[]> gaugesList, List<float[]> horizontalDeviationsList) {
        List<Float> tgiValues = new ArrayList<>();
        int satisfactoryInstances = 0;

        for (int i = 0; i < instances; i++) {
            double sigmaH = genH(HLeftList.get(i), HRightList.get(i));
            double sigmaS = genS(crossLevelsList.get(i), gaugesList.get(i), horizontalDeviationsList.get(i));

            float sigmaHFloat = (float) sigmaH;
            float sigmaSFloat = (float) sigmaS;

            float normalizedH = sigmaHFloat / Hlim;
            float normalizedS = 2 * (sigmaSFloat / Slim);

            float totalDeviation = normalizedH + normalizedS;
            int roundedDeviation = (int) Math.ceil(totalDeviation);
            float tgiOut = 150 - (100.0f / 3.0f) * roundedDeviation;

            tgiValues.add(tgiOut);

            if (sigmaH <= Hlim && sigmaS <= Slim) {
                satisfactoryInstances++;
            }
        }

        float satisfactoryLength = satisfactoryInstances * trackLength;
        float totalLength = instances * trackLength;
        float K = satisfactoryLength / totalLength;

        resultStage = new Stage();
        BorderPane resultPane = new BorderPane();
        resultPane.setPadding(new Insets(10));

        VBox resultBox = new VBox(10);
        resultBox.setPadding(new Insets(10));
        resultBox.setAlignment(Pos.CENTER_LEFT);

        resultBox.getChildren().add(new Label("QI Values: " + tgiValues.toString()));
        resultBox.getChildren().add(new Label("K Value: " + String.format("%.2f", K)));

        resultPane.setCenter(resultBox);

        Scene resultScene = new Scene(resultPane, 400, 400);
        resultStage.setScene(resultScene);
        resultStage.setTitle("Variation 5 Results");
        resultStage.show();
    }
    
    private void varTGIjCoeff (float SDz, float SDy, float SDw, float SDe) {
    	float fac = (float) (0.5 * SDe);
    	float num = SDz * SDy * SDw * fac;
        tgiOut = (float) (num / 3.5);
    	
        resultStage = new Stage();
        BorderPane resultPane = new BorderPane();
        resultPane.setPadding(new Insets(10));

        VBox resultBox = new VBox(10);
        resultBox.setPadding(new Insets(10));
        resultBox.setAlignment(Pos.CENTER_LEFT);

        resultBox.getChildren().add(new Label("J Output: " + tgiOut));

        resultPane.setCenter(resultBox);

        Scene resultScene = new Scene(resultPane, 400, 250);
        resultStage.setScene(resultScene);
        resultStage.setTitle("Results");
        resultStage.show();
    }
    
    private void varTGIcn(float facOne, float facTwo, float facThree, float facFour, float facFive, float facSix) {
        resultStage = new Stage();
        BorderPane resultPane = new BorderPane();
        resultPane.setPadding(new Insets(10));

        VBox resultBox = new VBox(10);
        resultBox.setPadding(new Insets(10));
        resultBox.setAlignment(Pos.CENTER_LEFT);

        String[] categories = {"Gauge", "Cross Level", "Left Alignment", "Right Alignment", "Left Surface", "Right Surface"};
        float[] stdvs = {facOne, facTwo, facThree, facFour, facFive, facSix};

        float[] tgiValues = new float[6];
        for (int i = 0; i < stdvs.length; i++) {
            tgiValues[i] = 1000 - 700 * (stdvs[i] * stdvs[i]);
        }

        float totalTGI = 0;
        for (float tgi : tgiValues) {
            totalTGI += tgi;
        }
        float averageTQI = totalTGI / tgiValues.length;

        for (int i = 0; i < categories.length; i++) {
            resultBox.getChildren().add(new Label(categories[i] + " TGI: " + String.format("%.2f", tgiValues[i])));
        }
        resultBox.getChildren().add(new Label("Overall TGI: " + String.format("%.2f", averageTQI)));

        resultPane.setCenter(resultBox);

        Scene resultScene = new Scene(resultPane, 400, 250);
        resultStage.setScene(resultScene);
        resultStage.setTitle("Results");
        resultStage.show();
    }
    
    private void varTGI(float SDu, float SDa, float SDt, float SDg, float SDuN, float SDaN, float SDtN, float SDgN,
            float SDuM, float SDaM, float SDtM, float SDgM) {
        resultStage = new Stage();
        BorderPane resultPane = new BorderPane();
        resultPane.setPadding(new Insets(10));

        VBox resultBox = new VBox(10);
        resultBox.setPadding(new Insets(10));
        resultBox.setAlignment(Pos.CENTER_LEFT);
        
        float uiFac = -1 * ((SDu - SDuN) / (SDuM - SDuN));
        float aiFac = -1 * ((SDa - SDaN) / (SDaM - SDaN));
        float tiFac = -1 * ((SDt - SDtN) / (SDtM - SDtN));
        float giFac = -1 * ((SDg - SDgN) / (SDgM - SDgN));  
        
        float ui = (float)(100 * (Math.exp(uiFac)));
        float ai = (float)(100 * (Math.exp(aiFac)));
        float ti = (float)(100 * (Math.exp(tiFac)));
        float gi = (float)(100 * (Math.exp(giFac)));
        
        float tgiNum = 2 * ui + ti + gi + 6 * ai;
        float TGI = tgiNum / 10;

        // Add results to the resultBox
        resultBox.getChildren().add(new Label("UI: " + String.format("%.2f", ui)));
        resultBox.getChildren().add(new Label("AI: " + String.format("%.2f", ai)));
        resultBox.getChildren().add(new Label("TI: " + String.format("%.2f", ti)));
        resultBox.getChildren().add(new Label("GI: " + String.format("%.2f", gi)));
        resultBox.getChildren().add(new Label("Overall TGI: " + String.format("%.2f", TGI)));

        resultPane.setCenter(resultBox);

        Scene resultScene = new Scene(resultPane, 400, 250);
        resultStage.setScene(resultScene);
        resultStage.setTitle("Results");
        resultStage.show();
    }
    
    private void varTGIvar(float L, float A, float G, float Lmax, float Amax, float Gmax) {
        resultStage = new Stage();
        BorderPane resultPane = new BorderPane();
        resultPane.setPadding(new Insets(10));

        VBox resultBox = new VBox(10);
        resultBox.setPadding(new Insets(10));
        resultBox.setAlignment(Pos.CENTER_LEFT);

        // Calculate each term, catching divide by zero errors
        float LFactor, GFactor, AFactor;

        try {
            LFactor = (L / Lmax) * 100;
        } catch (ArithmeticException e) {
            LFactor = 0; // If Lmax is 0, set factor to 0
        }

        try {
            GFactor = (G / Gmax) * 100;
        } catch (ArithmeticException e) {
            GFactor = 0; // If Gmax is 0, set factor to 0
        }

        try {
            AFactor = (A / Amax) * 100;
        } catch (ArithmeticException e) {
            AFactor = 0; // If Amax is 0, set factor to 0
        }

        // Calculate TGI
        float tgi = 100 - (1.0f / 3.0f) * (LFactor + GFactor + AFactor);

        // Clamp TGI to a minimum of 0
        tgi = Math.max(tgi, 0);

        // Determine TGI Classification
        String classification;
        if (tgi >= 80) {
            classification = "Excellent Condition";
        } else if (tgi >= 60) {
            classification = "Good Condition";
        } else if (tgi >= 40) {
            classification = "Fair Condition";
        } else if (tgi >= 20) {
            classification = "Poor Condition";
        } else {
            classification = "Very Poor Condition";
        }

        // Calculate exceedances (negative or zero values should show as 0)
        float lexceed = Math.max(L - Lmax, 0);
        float aexceed = Math.max(A - Amax, 0);
        float gexceed = Math.max(G - Gmax, 0);

        // Add results to resultBox
        resultBox.getChildren().add(new Label("TGI: " + String.format("%.2f", tgi)));
        resultBox.getChildren().add(new Label("TGI Classification: " + classification));
        resultBox.getChildren().add(new Label("L exceed: " + String.format("%.2f", lexceed)));
        resultBox.getChildren().add(new Label("A exceed: " + String.format("%.2f", aexceed)));
        resultBox.getChildren().add(new Label("G exceed: " + String.format("%.2f", gexceed)));

        resultPane.setCenter(resultBox);

        Scene resultScene = new Scene(resultPane, 400, 250);
        resultStage.setScene(resultScene);
        resultStage.setTitle("Results");
        resultStage.show();
    }
}