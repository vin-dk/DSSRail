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
   
 // Helper method to classify the TGI
    private String classifyTGI(float tgi) {
        if (tgi >= 80) {
            return "Excellent Condition";
        } else if (tgi >= 60) {
            return "Good Condition";
        } else if (tgi >= 40) {
            return "Fair Condition";
        } else if (tgi >= 20) {
            return "Poor Condition";
        } else {
            return "Very Poor Condition";
        }
    }
     
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
    
    private List<float[]> transposeData(List<float[]> data) {
    	//helper method, flips into arrays
        int rows = data.size();
        int cols = data.get(0).length;
        List<float[]> transposed = new ArrayList<>();

        for (int i = 0; i < cols; i++) {
            float[] newRow = new float[rows];
            for (int j = 0; j < rows; j++) {
                newRow[j] = data.get(j)[i];
            }
            transposed.add(newRow);
        }
        return transposed;
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
    
    public static double calc80(double[] values) {
        if (values == null || values.length == 0) {
            throw new IllegalArgumentException("Array must contain at least one element.");
        }

        Arrays.sort(values); 

        double index = 0.8 * (values.length - 1);  
        int lowerIndex = (int) Math.floor(index);
        int upperIndex = (int) Math.ceil(index);

        if (lowerIndex == upperIndex) {
            return values[lowerIndex];  
        } else {
            double weight = index - lowerIndex;
            return values[lowerIndex] * (1 - weight) + values[upperIndex] * weight;
        }
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
    
 // Helper method to convert List<Float> to float[]
    private float[] toPrimitiveArray(List<Float> list) {
        float[] array = new float[list.size()];
        for (int i = 0; i < list.size(); i++) {
            array[i] = list.get(i);
        }
        return array;
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
        Label classLabel = new Label("Class of Track: ");
        ComboBox<String> classCombo = new ComboBox<>();
        classCombo.getItems().addAll("1", "2", "3", "4", "5");

        Label typeLabel = new Label("Type of Track: ");
        ComboBox<String> typeCombo = new ComboBox<>();
        typeCombo.getItems().addAll("Line (Straight)", "31-foot Chord", "62-foot Chord", "31-foot Qualified Cant Chord", "62-foot Qualified Cant Chord");

        // File chooser button
        Label fileLabel = new Label("Select Excel File: ");
        Button fileButton = new Button("Browse...");
        Label selectedFileLabel = new Label("No file selected");
        
        FileChooser fileChooser = new FileChooser();
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Excel Files", "*.xlsx"));
        
        fileButton.setOnAction(e -> {
            File selectedFile = fileChooser.showOpenDialog(defaultWindow);
            if (selectedFile != null) {
                selectedFileLabel.setText(selectedFile.getAbsolutePath());
            }
        });

        // Add components to grid
        gridPane.add(notice, 0, 0);
        gridPane.add(classLabel, 0, 1);
        gridPane.add(classCombo, 1, 1);
        gridPane.add(typeLabel, 0, 2);
        gridPane.add(typeCombo, 1, 2);
        gridPane.add(fileLabel, 0, 3);
        gridPane.add(fileButton, 1, 3);
        gridPane.add(selectedFileLabel, 1, 4);

        pane.setCenter(gridPane);

        Button enterButton = new Button("Enter");
        enterButton.setOnAction(e -> {
            try {
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

                // Now that we have track class/type and limits, read Excel data
                if (!selectedFileLabel.getText().equals("No file selected")) {
                    File excelFile = new File(selectedFileLabel.getText());
                    List<float[]> longitudinalList = new ArrayList<>();
                    List<float[]> alignmentList = new ArrayList<>();
                    List<float[]> gaugeList = new ArrayList<>();

                    // Use Apache POI to read Excel
                    try (FileInputStream fis = new FileInputStream(excelFile);
                         Workbook workbook = new XSSFWorkbook(fis)) {
                        Sheet sheet = workbook.getSheetAt(0);

                        for (Row row : sheet) {
                            // Read each row, expecting 3 columns: L, A, G
                            float l = (float) row.getCell(0).getNumericCellValue();
                            float a = (float) row.getCell(1).getNumericCellValue();
                            float g = (float) row.getCell(2).getNumericCellValue();

                            longitudinalList.add(new float[]{l});
                            alignmentList.add(new float[]{a});
                            gaugeList.add(new float[]{g});
                        }
                    } catch (IOException | NullPointerException ex) {
                        showError("Error reading Excel file or invalid format. Ensure proper L, A, G values. Refer to info panel"
                        		+ " for help");
                    }

                    // After collecting data, perform calculations
                    int instances = longitudinalList.size();  // Number of rows = number of instances
                    varDefaultTGI(instances, Lmax, Amax, Gmax, longitudinalList, alignmentList, gaugeList);
                    defaultWindow.close();
                } else {
                    showError("Please select an Excel file.");
                }
            } catch (Exception ex) {
                showError("Invalid input. Please enter valid numbers.");
            }
        });

        BorderPane bottomPane = new BorderPane();
        bottomPane.setRight(enterButton);
        BorderPane.setMargin(enterButton, new Insets(10));
        pane.setBottom(bottomPane);

        Scene scene = new Scene(pane, 400, 400);
        defaultWindow.setScene(scene);
        defaultWindow.setTitle("Default TGI Input");
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

        Label notice = new Label("All measurements in inches");
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

        Label notice = new Label("All measurements in inches");
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

        Label notice = new Label("All measurements in inches");
        notice.setStyle("-fx-font-weight: bold;");

        Label fileLabel = new Label("Select Excel File: ");
        Button fileButton = new Button("Browse...");
        Label selectedFileLabel = new Label("No file selected");
        
        FileChooser fileChooser = new FileChooser();
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Excel Files", "*.xlsx"));

        fileButton.setOnAction(e -> {
            File selectedFile = fileChooser.showOpenDialog(NTQIWindow);
            if (selectedFile != null) {
                selectedFileLabel.setText(selectedFile.getAbsolutePath());
            }
        });

        // Add components to grid
        gridPane.add(notice, 0, 0);
        gridPane.add(fileLabel, 0, 1);
        gridPane.add(fileButton, 1, 1);
        gridPane.add(selectedFileLabel, 1, 2);

        pane.setCenter(gridPane);

        Button enterButton = new Button("Enter");
        enterButton.setOnAction(e -> {
            try {
                // Validate file selection
                if (!selectedFileLabel.getText().equals("No file selected")) {
                    File excelFile = new File(selectedFileLabel.getText());
                    List<float[]> segmentDataList = new ArrayList<>();

                    // Use Apache POI to read Excel
                    try (FileInputStream fis = new FileInputStream(excelFile);
                         Workbook workbook = new XSSFWorkbook(fis)) {
                        Sheet sheet = workbook.getSheetAt(0);

                        int rowCount = sheet.getPhysicalNumberOfRows();
                        int colCount = sheet.getRow(0).getPhysicalNumberOfCells();

                        // Ensure square matrix
                        for (Row row : sheet) {
                            if (row.getPhysicalNumberOfCells() != colCount) {
                                throw new IllegalArgumentException("Each row must have the same number of columns.");
                            }
                            float[] segmentValues = new float[colCount];
                            for (int i = 0; i < colCount; i++) {
                                segmentValues[i] = (float) row.getCell(i).getNumericCellValue();
                            }
                            segmentDataList.add(segmentValues);
                        }

                        // Transpose the data so each column is a segment
                        List<float[]> transposedData = transposeData(segmentDataList);

                        // Calculate standard deviation for each segment
                        List<Double> stdDevs = new ArrayList<>();
                        for (float[] segment : transposedData) {
                            stdDevs.add(stdv(segment));
                        }

                        // Get the 80th percentile of the standard deviations
                        double[] stdDevArray = stdDevs.stream().mapToDouble(Double::doubleValue).toArray();
                        float[] stdDevFloatArray = new float[stdDevArray.length];
                        for (int i = 0; i < stdDevArray.length; i++) {
                            stdDevFloatArray[i] = (float) stdDevArray[i];
                        }
                        double percentile80 = calc80(stdDevArray);

                        // Output scores for each segment
                        varTGIntqi(stdDevs, percentile80);

                    } catch (IOException | NullPointerException ex) {
                        showError("Error reading Excel file or invalid format. Refer to info panel for help.");
                    }
                } else {
                    showError("Please select an Excel file.");
                }
            } catch (Exception ex) {
                showError("Invalid input. Please enter valid numbers.");
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

        Label notice = new Label("All measurements in inches");
        notice.setStyle("-fx-font-weight: bold;");

        Label classLabel = new Label("Track Class: ");
        ComboBox<String> classCombo = new ComboBox<>();
        classCombo.getItems().addAll("1", "2", "3", "4", "5");

        Label fileLabel = new Label("Select Excel File: ");
        Button fileButton = new Button("Browse...");
        Label selectedFileLabel = new Label("No file selected");

        FileChooser fileChooser = new FileChooser();
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Excel Files", "*.xlsx"));

        fileButton.setOnAction(e -> {
            File selectedFile = fileChooser.showOpenDialog(swedenQWindow);
            if (selectedFile != null) {
                selectedFileLabel.setText(selectedFile.getAbsolutePath());
            }
        });

        Button nextButton = new Button("Next");

        nextButton.setOnAction(e -> {
            try {
                int trackClass = Integer.parseInt(classCombo.getValue());
                float[] limits = getTrackClassLimits(trackClass); // Get Hlim and Slim based on class

                if (!selectedFileLabel.getText().equals("No file selected")) {
                    File excelFile = new File(selectedFileLabel.getText());

                    List<float[]> HLeftList = new ArrayList<>();
                    List<float[]> HRightList = new ArrayList<>();
                    List<float[]> crossLevelsList = new ArrayList<>();
                    List<float[]> gaugesList = new ArrayList<>();
                    List<float[]> horizontalDeviationsList = new ArrayList<>();

                    // Use Apache POI to read Excel
                    try (FileInputStream fis = new FileInputStream(excelFile);
                         Workbook workbook = new XSSFWorkbook(fis)) {
                        Sheet sheet = workbook.getSheetAt(0);

                        // Ensure the data is rectangular and correctly separated into instances
                        int totalColumns = sheet.getRow(0).getLastCellNum();
                        if (totalColumns % 5 != 0) {
                            throw new IllegalArgumentException("Invalid format: Each instance must have 5 columns.");
                        }

                        int rowCount = sheet.getPhysicalNumberOfRows();
                        for (int i = 0; i < totalColumns; i += 5) {  // Jump by 5 columns for each instance
                            List<Float> hLeft = new ArrayList<>();
                            List<Float> hRight = new ArrayList<>();
                            List<Float> crossLevels = new ArrayList<>();
                            List<Float> gauges = new ArrayList<>();
                            List<Float> horizontalDeviations = new ArrayList<>();

                            for (Row row : sheet) {
                                // Ensure no missing values in the block
                                for (int j = i; j < i + 5; j++) {
                                    if (row.getCell(j) == null || row.getCell(j).getCellType() != CellType.NUMERIC) {
                                        throw new IllegalArgumentException("Invalid format: Missing or non-numeric value detected.");
                                    }
                                }

                                // Collect the data from each column in the current 5-column block
                                hLeft.add((float) row.getCell(i).getNumericCellValue());
                                hRight.add((float) row.getCell(i + 1).getNumericCellValue());
                                crossLevels.add((float) row.getCell(i + 2).getNumericCellValue());
                                gauges.add((float) row.getCell(i + 3).getNumericCellValue());
                                horizontalDeviations.add((float) row.getCell(i + 4).getNumericCellValue());
                            }

                            // Add data for the current instance
                            HLeftList.add(toPrimitiveArray(hLeft));
                            HRightList.add(toPrimitiveArray(hRight));
                            crossLevelsList.add(toPrimitiveArray(crossLevels));
                            gaugesList.add(toPrimitiveArray(gauges));
                            horizontalDeviationsList.add(toPrimitiveArray(horizontalDeviations));
                        }
                    } catch (IOException | NullPointerException ex) {
                        showError("Error reading Excel file or invalid format. Ensure proper data. Click info for help.");
                    }

                    // Pass the data to varTGIswedenQ for processing
                    varTGIswedenQ(HLeftList.size(), limits[0], limits[1], HLeftList, HRightList, crossLevelsList, gaugesList, horizontalDeviationsList);
                    swedenQWindow.close();
                } else {
                    showError("Please select an Excel file.");
                }
            } catch (Exception ex) {
                showError("Invalid input. Please enter valid selections. Click info for help.");
            }
        });

        gridPane.add(notice, 0, 0);
        gridPane.add(classLabel, 0, 1);
        gridPane.add(classCombo, 1, 1);
        gridPane.add(fileLabel, 0, 2);
        gridPane.add(fileButton, 1, 2);
        gridPane.add(selectedFileLabel, 1, 3);
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

        Label notice = new Label("All measurements in inches");
        notice.setStyle("-fx-font-weight: bold;");

        Label fileLabel = new Label("Select Excel File: ");
        Button fileButton = new Button("Browse...");
        Label selectedFileLabel = new Label("No file selected");

        FileChooser fileChooser = new FileChooser();
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Excel Files", "*.xlsx"));

        fileButton.setOnAction(e -> {
            File selectedFile = fileChooser.showOpenDialog(jCoeffWindow);
            if (selectedFile != null) {
                selectedFileLabel.setText(selectedFile.getAbsolutePath());
            }
        });

        gridPane.add(notice, 0, 0);
        gridPane.add(fileLabel, 0, 1);
        gridPane.add(fileButton, 1, 1);
        gridPane.add(selectedFileLabel, 1, 2);

        pane.setCenter(gridPane);

        Button enterButton = new Button("Enter");
        enterButton.setOnAction(e -> {
            if (!selectedFileLabel.getText().equals("No file selected")) {
                File excelFile = new File(selectedFileLabel.getText());

                try (FileInputStream fis = new FileInputStream(excelFile);
                     Workbook workbook = new XSSFWorkbook(fis)) {
                    
                    Sheet sheet = workbook.getSheetAt(0);

                    if (sheet.getRow(0).getLastCellNum() != 4) {
                        throw new IllegalArgumentException("Excel file must have exactly 4 columns.");
                    }

                    List<Float> Z = new ArrayList<>();
                    List<Float> Y = new ArrayList<>();
                    List<Float> W = new ArrayList<>();
                    List<Float> E = new ArrayList<>();

                    for (Row row : sheet) {
                        if (row.getPhysicalNumberOfCells() < 4) {
                            throw new IllegalArgumentException("Each row must have exactly 4 values.");
                        }

                        Z.add((float) row.getCell(0).getNumericCellValue());
                        Y.add((float) row.getCell(1).getNumericCellValue());
                        W.add((float) row.getCell(2).getNumericCellValue());
                        E.add((float) row.getCell(3).getNumericCellValue());
                    }

                    float[] ZArray = toPrimitiveArray(Z);
                    float[] YArray = toPrimitiveArray(Y);
                    float[] WArray = toPrimitiveArray(W);
                    float[] EArray = toPrimitiveArray(E);

                    float SDz = (float) stdv(ZArray);
                    float SDy = (float) stdv(YArray);
                    float SDw = (float) stdv(WArray);
                    float SDe = (float) stdv(EArray);

                    varTGIjCoeff(SDz, SDy, SDw, SDe);
                    jCoeffWindow.close();

                } catch (IOException | IllegalArgumentException ex) {
                    showError("Error reading Excel file or invalid format. Ensure the file has exactly 4 numeric columns with equal lengths. Press info for help.");
                }
            } else {
                showError("Please select an Excel file.");
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
    
    private void openCNWindow() {
        Stage CNWindow = new Stage();
        BorderPane pane = new BorderPane();
        pane.setPadding(new Insets(10));

        GridPane gridPane = new GridPane();
        gridPane.setPadding(new Insets(10));
        gridPane.setHgap(10);
        gridPane.setVgap(10);
        gridPane.setAlignment(Pos.CENTER_LEFT);

        Label notice = new Label("All measurements in inches");
        notice.setStyle("-fx-font-weight: bold;");

        Label fileLabel = new Label("Select Excel File: ");
        Button fileButton = new Button("Browse...");
        Label selectedFileLabel = new Label("No file selected");

        FileChooser fileChooser = new FileChooser();
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Excel Files", "*.xlsx"));

        fileButton.setOnAction(e -> {
            File selectedFile = fileChooser.showOpenDialog(CNWindow);
            if (selectedFile != null) {
                selectedFileLabel.setText(selectedFile.getAbsolutePath());
            }
        });

        gridPane.add(notice, 0, 0);
        gridPane.add(fileLabel, 0, 1);
        gridPane.add(fileButton, 1, 1);
        gridPane.add(selectedFileLabel, 1, 2);

        pane.setCenter(gridPane);

        Button enterButton = new Button("Enter");
        enterButton.setOnAction(e -> {
            if (!selectedFileLabel.getText().equals("No file selected")) {
                File excelFile = new File(selectedFileLabel.getText());

                try (FileInputStream fis = new FileInputStream(excelFile);
                     Workbook workbook = new XSSFWorkbook(fis)) {
                    
                    Sheet sheet = workbook.getSheetAt(0);

                    // Ensure the data has exactly 6 columns
                    if (sheet.getRow(0).getLastCellNum() != 6) {
                        throw new IllegalArgumentException("Excel file must have exactly 6 columns.");
                    }

                    List<Float> gauges = new ArrayList<>();
                    List<Float> crossLevels = new ArrayList<>();
                    List<Float> leftSurfaces = new ArrayList<>();
                    List<Float> rightSurfaces = new ArrayList<>();
                    List<Float> leftAlignments = new ArrayList<>();
                    List<Float> rightAlignments = new ArrayList<>();

                    int rowCount = sheet.getPhysicalNumberOfRows();
                    for (Row row : sheet) {
                        if (row.getPhysicalNumberOfCells() < 6) {
                            throw new IllegalArgumentException("Each row must contain exactly 6 numeric values.");
                        }

                        gauges.add((float) row.getCell(0).getNumericCellValue());
                        crossLevels.add((float) row.getCell(1).getNumericCellValue());
                        leftSurfaces.add((float) row.getCell(2).getNumericCellValue());
                        rightSurfaces.add((float) row.getCell(3).getNumericCellValue());
                        leftAlignments.add((float) row.getCell(4).getNumericCellValue());
                        rightAlignments.add((float) row.getCell(5).getNumericCellValue());
                    }

                    float[] gaugeArray = toPrimitiveArray(gauges);
                    float[] crossLevelArray = toPrimitiveArray(crossLevels);
                    float[] leftSurfaceArray = toPrimitiveArray(leftSurfaces);
                    float[] rightSurfaceArray = toPrimitiveArray(rightSurfaces);
                    float[] leftAlignmentArray = toPrimitiveArray(leftAlignments);
                    float[] rightAlignmentArray = toPrimitiveArray(rightAlignments);

                    float stdvGauge = (float) stdv(gaugeArray);
                    float stdvCross = (float) stdv(crossLevelArray);
                    float stdvLeftS = (float) stdv(leftSurfaceArray);
                    float stdvRightS = (float) stdv(rightSurfaceArray);
                    float stdvLeftA = (float) stdv(leftAlignmentArray);
                    float stdvRightA = (float) stdv(rightAlignmentArray);

                    varTGIcn(stdvGauge, stdvCross, stdvLeftA, stdvRightA, stdvLeftS, stdvRightS);
                    CNWindow.close();

                } catch (IOException | IllegalArgumentException ex) {
                    showError("Error reading Excel file or invalid format. Ensure the file has exactly 6 numeric columns with equal lengths. Press info for help.");
                }
            } else {
                showError("Please select an Excel file.");
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

        ToggleGroup speedGroup = new ToggleGroup();
        RadioButton above105 = new RadioButton("> 105 kph");
        above105.setToggleGroup(speedGroup);
        RadioButton below105 = new RadioButton("< 105 kph");
        below105.setToggleGroup(speedGroup);
        
        Label fileLabel = new Label("Select Excel File: ");
        Button fileButton = new Button("Browse...");
        Label selectedFileLabel = new Label("No file selected");

        FileChooser fileChooser = new FileChooser();
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Excel Files", "*.xlsx"));

        fileButton.setOnAction(e -> {
            File selectedFile = fileChooser.showOpenDialog(TGIWindow);
            if (selectedFile != null) {
                selectedFileLabel.setText(selectedFile.getAbsolutePath());
            }
        });

        gridPane.add(notice, 0, 0);
        gridPane.add(above105, 0, 1);
        gridPane.add(below105, 0, 2);
        gridPane.add(fileLabel, 0, 3);
        gridPane.add(fileButton, 1, 3);
        gridPane.add(selectedFileLabel, 1, 4);

        pane.setCenter(gridPane);

        Button enterButton = new Button("Enter");
        enterButton.setOnAction(e -> {
            try {
                if (!selectedFileLabel.getText().equals("No file selected")) {
                    File excelFile = new File(selectedFileLabel.getText());

                    List<Float> longitudinalLevelList = new ArrayList<>();
                    List<Float> alignmentList = new ArrayList<>();
                    List<Float> gaugeList = new ArrayList<>();
                    List<Float> twistList = new ArrayList<>();

                    try (FileInputStream fis = new FileInputStream(excelFile);
                         Workbook workbook = new XSSFWorkbook(fis)) {
                        Sheet sheet = workbook.getSheetAt(0);

                        if (sheet.getRow(0).getLastCellNum() != 4) {
                            throw new IllegalArgumentException("Excel file must have exactly 4 columns.");
                        }

                        for (Row row : sheet) {
                            if (row.getPhysicalNumberOfCells() < 4) {
                                throw new IllegalArgumentException("Each row must contain exactly 4 numeric values.");
                            }

                            longitudinalLevelList.add((float) row.getCell(0).getNumericCellValue());
                            alignmentList.add((float) row.getCell(1).getNumericCellValue());
                            gaugeList.add((float) row.getCell(2).getNumericCellValue());
                            twistList.add((float) row.getCell(3).getNumericCellValue());
                        }

                        float[] longitudinalLevel = toPrimitiveArray(longitudinalLevelList);
                        float[] alignment = toPrimitiveArray(alignmentList);
                        float[] gauge = toPrimitiveArray(gaugeList);
                        float[] twist = toPrimitiveArray(twistList);

                        float SDu = (float) stdv(longitudinalLevel);
                        float SDa = (float) stdv(alignment);
                        float SDg = (float) stdv(gauge);
                        float SDt = (float) stdv(twist);

                        float SDnewLong = 2.5f;
                        float SDnewAlign = 1.5f;
                        float SDnewGauge = 1.0f;
                        float SDnewTwist = 1.75f;

                        float SDmainLong = above105.isSelected() ? 6.2f : 7.2f;
                        float SDmainAlign = 3.0f;
                        float SDmainGauge = 3.6f;
                        float SDmainTwist = above105.isSelected() ? 3.8f : 4.2f;

                        varTGI(SDu, SDa, SDt, SDg, SDnewLong, SDnewAlign, SDnewTwist, SDnewGauge, SDmainLong, SDmainAlign, SDmainTwist, SDmainGauge);
                        TGIWindow.close();

                    } catch (IOException | IllegalArgumentException ex) {
                        showError("Error reading Excel file or invalid format. Ensure the file has exactly 4 numeric columns with equal lengths. Press info for help.");
                    }
                } else {
                    showError("Please select an Excel file.");
                }
            } catch (Exception ex) {
                showError("Invalid input. Please enter valid selections.");
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

        Label fileLabel = new Label("Select Excel File: ");
        Button fileButton = new Button("Browse...");
        Label selectedFileLabel = new Label("No file selected");

        FileChooser fileChooser = new FileChooser();
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Excel Files", "*.xlsx"));

        fileButton.setOnAction(e -> {
            File selectedFile = fileChooser.showOpenDialog(TGIVarWindow);
            if (selectedFile != null) {
                selectedFileLabel.setText(selectedFile.getAbsolutePath());
            }
        });

        gridPane.add(notice, 0, 0);
        gridPane.add(classLabel, 0, 1);
        gridPane.add(classCombo, 1, 1);
        gridPane.add(typeLabel, 0, 2);
        gridPane.add(typeCombo, 1, 2);
        gridPane.add(fileLabel, 0, 3);
        gridPane.add(fileButton, 1, 3);
        gridPane.add(selectedFileLabel, 1, 4);

        pane.setCenter(gridPane);

        Button enterButton = new Button("Enter");
        enterButton.setOnAction(e -> {
            try {
                int trackClass = Integer.parseInt(classCombo.getValue());
                String trackType = typeCombo.getValue();

                // Get max allowable deviations based on class and type
                float Lmax = 0, Gmax = 0, Amax = 0;

                switch (trackType) {
                    case "Line (Straight)" -> {
                        switch (trackClass) {
                            case 1 -> { Lmax = 3; Gmax = 1; Amax = 5; }
                            case 2 -> { Lmax = 2; Gmax = 0.875f; Amax = 3; }
                            case 3 -> { Lmax = 1.75f; Gmax = 0.875f; Amax = 1.75f; }
                            case 4 -> { Lmax = 1.5f; Gmax = 0.75f; Amax = 1.5f; }
                            case 5 -> { Lmax = 1; Gmax = 0.75f; Amax = 0.75f; }
                        }
                    }
                    case "31-foot Chord" -> {
                        switch (trackClass) {
                            case 1, 2 -> { Lmax = 0; Gmax = 1; Amax = 0; } // N/A for L and A
                            case 3 -> { Lmax = 1; Gmax = 0.875f; Amax = 1.25f; }
                            case 4 -> { Lmax = 1; Gmax = 0.75f; Amax = 1; }
                            case 5 -> { Lmax = 1; Gmax = 0.75f; Amax = 0.5f; }
                        }
                    }
                    case "62-foot Chord" -> {
                        switch (trackClass) {
                            case 1 -> { Lmax = 3; Gmax = 1; Amax = 5; }
                            case 2 -> { Lmax = 2.75f; Gmax = 0.875f; Amax = 3; }
                            case 3 -> { Lmax = 2.25f; Gmax = 0.875f; Amax = 1.75f; }
                            case 4 -> { Lmax = 2; Gmax = 0.75f; Amax = 1.5f; }
                            case 5 -> { Lmax = 1.25f; Gmax = 0.75f; Amax = 0.625f; }
                        }
                    }
                    case "31-foot Qualified Cant Chord", "62-foot Qualified Cant Chord" -> {
                        switch (trackClass) {
                            case 1 -> { Lmax = 0; Gmax = 1; Amax = 0; } // N/A for L and A
                            case 2 -> { Lmax = 0; Gmax = 0.875f; Amax = 0; } // N/A for L and A
                            case 3 -> { Lmax = 1; Gmax = 0.875f; Amax = 1.25f; }
                            case 4 -> { Lmax = 1; Gmax = 0.75f; Amax = 1; }
                            case 5 -> { Lmax = 1; Gmax = 0.75f; Amax = 0.5f; }
                        }
                    }
                }

                if (!selectedFileLabel.getText().equals("No file selected")) {
                    File excelFile = new File(selectedFileLabel.getText());

                    List<Float[]> instances = new ArrayList<>();

                    // Use Apache POI to read Excel
                    try (FileInputStream fis = new FileInputStream(excelFile);
                         Workbook workbook = new XSSFWorkbook(fis)) {
                        Sheet sheet = workbook.getSheetAt(0);

                        // Ensure the Excel file has exactly 3 columns (L, A, G)
                        if (sheet.getRow(0).getLastCellNum() != 3) {
                            throw new IllegalArgumentException("Excel file must have exactly 3 columns (L, A, G).");
                        }

                        // Read values for each instance (each row)
                        for (Row row : sheet) {
                            if (row.getPhysicalNumberOfCells() < 3) {
                                throw new IllegalArgumentException("Each row must contain exactly 3 numeric values.");
                            }

                            Float[] instance = new Float[3];
                            instance[0] = (float) row.getCell(0).getNumericCellValue();  // L
                            instance[1] = (float) row.getCell(1).getNumericCellValue();  // A
                            instance[2] = (float) row.getCell(2).getNumericCellValue();  // G

                            instances.add(instance);
                        }

                        // Pass data to varTGIvar to calculate and display results
                        varTGIvar(instances, Lmax, Amax, Gmax);
                        TGIVarWindow.close();

                    } catch (IOException | IllegalArgumentException ex) {
                        showError("Error reading Excel file or invalid format. Ensure the file has exactly 3 numeric columns.");
                    }
                } else {
                    showError("Please select an Excel file.");
                }
            } catch (Exception ex) {
                showError("Invalid input. Please enter valid selections.");
            }
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
    
    private void varTGIntqi(List<Double> stdDevs, double percentile80) {
        resultStage = new Stage();
        BorderPane resultPane = new BorderPane();
        resultPane.setPadding(new Insets(10));

        VBox resultBox = new VBox(10);
        resultBox.setPadding(new Insets(10));
        resultBox.setAlignment(Pos.CENTER_LEFT);

        // Output TGI score for each segment
        for (int i = 0; i < stdDevs.size(); i++) {
            double stddev = stdDevs.get(i);
            double factor = stddev / percentile80;
            float tgiOut = (float) (10 * Math.pow(0.675, factor));

            resultBox.getChildren().add(new Label("Segment " + (i + 1) + " TGI Output: " + tgiOut));
        }

        resultPane.setCenter(resultBox);

        Scene resultScene = new Scene(resultPane, 400, 400);
        resultStage.setScene(resultScene);
        resultStage.setTitle("Results");
        resultStage.show();
    }
   
    private void varTGIswedenQ(int instances, float Hlim, float Slim, List<float[]> HLeftList, List<float[]> HRightList, List<float[]> crossLevelsList, List<float[]> gaugesList, List<float[]> horizontalDeviationsList) {
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

        float satisfactoryLength = satisfactoryInstances * instances; // Use instances count to represent length
        float K = satisfactoryLength / instances;

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
        resultStage.setTitle("Sweden Q Results");
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
    
    private void varTGIvar(List<Float[]> instances, float Lmax, float Amax, float Gmax) {
        int satisfactoryCount = 0;  
        List<String> results = new ArrayList<>();  

        for (int i = 0; i < instances.size(); i++) {
            Float[] instance = instances.get(i);
            float L = instance[0], A = instance[1], G = instance[2];

            float LFactor = (Lmax == 0) ? 0 : (L / Lmax) * 100;
            float GFactor = (Gmax == 0) ? 0 : (G / Gmax) * 100;
            float AFactor = (Amax == 0) ? 0 : (A / Amax) * 100;

            float tgi = 100 - (1.0f / 3.0f) * (LFactor + GFactor + AFactor);
            tgi = Math.max(tgi, 0); 

            String classification = classifyTGI(tgi);

            float lexceed = Math.max(L - Lmax, 0);
            float aexceed = Math.max(A - Amax, 0);
            float gexceed = Math.max(G - Gmax, 0);

            if (lexceed == 0 && aexceed == 0 && gexceed == 0) {
                satisfactoryCount++;
            }

            results.add("Instance " + (i + 1) + ": TGI = " + String.format("%.2f", tgi) 
                        + ", Classification = " + classification 
                        + ", L exceed: " + String.format("%.2f", lexceed) 
                        + ", A exceed: " + String.format("%.2f", aexceed) 
                        + ", G exceed: " + String.format("%.2f", gexceed));
        }

        float K = (float) satisfactoryCount / instances.size();

        resultStage = new Stage();
        BorderPane resultPane = new BorderPane();
        resultPane.setPadding(new Insets(10));

        VBox resultBox = new VBox(10);
        resultBox.setPadding(new Insets(10));
        resultBox.setAlignment(Pos.CENTER_LEFT);

        for (String result : results) {
            resultBox.getChildren().add(new Label(result));
        }

        resultBox.getChildren().add(new Label("K-Value: " + String.format("%.2f", K)));

        resultPane.setCenter(resultBox);

        Scene resultScene = new Scene(resultPane, 400, 400);
        resultStage.setScene(resultScene);
        resultStage.setTitle("TGI Variation Results");
        resultStage.show();
    }
}