import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.RadioButton;
import javafx.scene.control.ScrollPane;
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

import javafx.stage.DirectoryChooser;
import javafx.stage.FileChooser;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Scanner;
import javafx.scene.layout.HBox; 
import java.text.SimpleDateFormat;
import java.util.Date;


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
    
 // Helper method to load text from a .txt file in the resources folder
    private String loadTextFromResource(String resourcePath) {
        try (InputStream inputStream = getClass().getResourceAsStream(resourcePath);
             Scanner scanner = new Scanner(inputStream, StandardCharsets.UTF_8.name())) {
            // Read the entire file content
            return scanner.useDelimiter("\\A").next();
        } catch (Exception e) {
            e.printStackTrace();
            return "Failed to load text.";
        }
    }
    
    private void showInfoPopup() {
        Stage infoStage = new Stage();
        VBox vbox = new VBox(10);
        vbox.setPadding(new Insets(10));

        Image image = new Image(getClass().getResourceAsStream("/DefaultEx.png"));
        ImageView imageView = new ImageView(image);
        imageView.setPreserveRatio(true);

        double imageOriginalWidth = image.getWidth();
        imageView.setFitWidth(imageOriginalWidth / 2);  
        imageView.setPreserveRatio(true);  

        String textContent = loadTextFromResource("/DefaultEx.txt");
        Label infoText = new Label(textContent);
        infoText.setWrapText(true);

        vbox.getChildren().addAll(imageView, infoText);

        ScrollPane scrollPane = new ScrollPane(vbox);
        scrollPane.setFitToWidth(true); 
        scrollPane.setFitToHeight(true); 

        Scene scene = new Scene(scrollPane, 800, 625);
        infoStage.setScene(scene);

        infoStage.setResizable(true); 
        infoStage.sizeToScene(); 
        infoStage.setTitle("Info");
        infoStage.show();
    }
    
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

    private float[] getTrackClassLimits(String trackClass) {
        switch (trackClass) {
            case "K0": return new float[] {1.1f, 1.6f};
            case "K1": return new float[] {1.3f, 1.7f};
            case "K2": return new float[] {1.5f, 1.9f};
            case "K3": return new float[] {1.9f, 2.4f};
            case "K4": return new float[] {2.4f, 3.1f};
            case "K5": return new float[] {2.9f, 3.6f};
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
    
    private String getFilePath() {
        String homeDir = System.getProperty("user.home");
        File resourceDir = new File(homeDir, "TGIProgramResources");
        File filePathFile = new File(resourceDir, "file_path.txt");

        try (BufferedReader reader = new BufferedReader(new FileReader(filePathFile))) {
            return reader.readLine();  // Return the saved path to the TGI Results directory
        } catch (IOException e) {
            e.printStackTrace();
            showError("Failed to retrieve file path. Please set up the TGI Results directory.");
            return null;
        }
    }
    
    private boolean isFilePathValid() {
        String homeDir = System.getProperty("user.home");
        File resourceDir = new File(homeDir, "TGIProgramResources");
        File filePathFile = new File(resourceDir, "file_path.txt");

        // Check if file_path.txt is present and points to a valid directory
        if (filePathFile.exists()) {
            try (BufferedReader reader = new BufferedReader(new FileReader(filePathFile))) {
                String filePath = reader.readLine();
                File tgiResultsDir = new File(filePath);
                if (tgiResultsDir.exists() && tgiResultsDir.isDirectory()) {
                    return true;  // Path is valid
                } else {
                    // Path is invalid, so delete file_path.txt
                    filePathFile.delete();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return false;  // Either file_path.txt doesn't exist, or it points to a non-existent directory
    }
    
 // Helper method to save the valid directory path to file_path.txt
    private void saveFilePath(String resultDirectoryPath) {
        String homeDir = System.getProperty("user.home");
        File resourceDir = new File(homeDir, "TGIProgramResources");
        if (!resourceDir.exists()) {
            resourceDir.mkdir();  // Create TGIProgramResources folder in the user's home directory if it doesn't exist
        }
        File filePathFile = new File(resourceDir, "file_path.txt");
        
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePathFile))) {
            writer.write(resultDirectoryPath);  // Write the path to file_path.txt
        } catch (IOException e) {
            e.printStackTrace();
            showError("Failed to save file path.");
        }
    }
    
    private void showInfo(String message) {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle("Information");
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }
    
 // The setupFile method saves in the user’s home directory
    private void setupFile() {
        DirectoryChooser directoryChooser = new DirectoryChooser();
        directoryChooser.setTitle("Select Directory for TGI Results");
        File selectedDirectory = directoryChooser.showDialog(new Stage());

        if (selectedDirectory != null) {
            String resultDirectoryPath = selectedDirectory.getAbsolutePath() + "/TGI Results";
            File resultsDirectory = new File(resultDirectoryPath);
            if (!resultsDirectory.exists()) {
                resultsDirectory.mkdir();  // Create TGI Results directory if it doesn't exist
            }

            // Save the directory path to file_path.txt in the user's home directory
            saveFilePath(resultDirectoryPath);

            // Create 9 empty Excel sheets in the TGI Results folder, labeled according to the methods
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
                        optionName = "Variation 3";
                        break;
                    case 5:
                        optionName = "Track Geometry Index";
                        break;
                    case 6:
                        optionName = "Netherlands Track Quality Index";
                        break;
                    case 7:
                        optionName = "Sweden Q";
                        break;
                    case 8:
                        optionName = "J Coefficient";
                        break;
                    case 9:
                        optionName = "CN Index";
                        break;
                    default:
                        optionName = "Default";
                }

                // Create an empty Excel sheet for each method
                File excelFile = new File(resultDirectoryPath + "/" + optionName + ".xlsx");
                try (Workbook workbook = new XSSFWorkbook()) {
                    Sheet sheet = workbook.createSheet(optionName);
                    try (FileOutputStream fos = new FileOutputStream(excelFile)) {
                        workbook.write(fos);
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                    showError("Failed to create Excel file for: " + optionName);
                }
            }

            showInfo("TGI Results folder set up successfully at: " + resultDirectoryPath);
        } else {
            showError("No directory selected. Setup failed.");
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
        
        String filePathResource = "/resources/file_path.txt";
        boolean isFilePathValid = isFilePathValid();  // Check if file_path.txt exists and is valid

        // Step 2: If file_path.txt is not valid, call setupFile method
        if (!isFilePathValid) {
            setupFile();
            openMenuSelection(primaryStage);
            return;  // Exit after setup, since the user will restart
        }

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
                    optionName = "Variation 3";
                    break;
                case 5:
                    optionName = "Track Geometry Index";
                    break;
                case 6:
                    optionName = "Netherlands Track Quality Index";
                    break;
                case 7:
                    optionName = "Sweden Q";
                    break;
                case 8:
                    optionName = "J Coefficient";
                    break;
                case 9:
                    optionName = "CN Index";
                    break;
                default:
                    optionName = "Default";
            }

            final String selectedOption = optionName;
            HBox optionBox = new HBox(5); 

            RadioButton variantOption = new RadioButton(optionName);
            variantOption.setToggleGroup(toggleGroup);
            variantOption.setOnAction(e -> userChoice = selectedOption);

            Button helpButton = new Button("?");
            helpButton.setStyle(
                "-fx-background-color: lightgray; " +   
                "-fx-text-fill: black; " +              
                "-fx-font-weight: bold; " +             
                "-fx-background-radius: 15; " +         
                "-fx-padding: 2; " +                    
                "-fx-min-width: 20px; " +               
                "-fx-min-height: 20px; " +              
                "-fx-max-width: 20px; " +               
                "-fx-max-height: 20px;"                 
            );
            helpButton.setOnAction(e -> showInfoPopup());

            optionBox.getChildren().addAll(variantOption, helpButton);
            optionsBox.getChildren().add(optionBox);
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
                case "Variation 3":
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

        Scene scene = new Scene(root);
        primaryStage.setScene(scene);
        primaryStage.sizeToScene();
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

                if (!selectedFileLabel.getText().equals("No file selected")) {
                    File excelFile = new File(selectedFileLabel.getText());
                    List<float[]> longitudinalList = new ArrayList<>();
                    List<float[]> alignmentList = new ArrayList<>();
                    List<float[]> gaugeList = new ArrayList<>();

                    try (FileInputStream fis = new FileInputStream(excelFile);
                         Workbook workbook = new XSSFWorkbook(fis)) {
                        Sheet sheet = workbook.getSheetAt(0);

                        for (Row row : sheet) {
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

                    int instances = longitudinalList.size();  
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

        Scene scene = new Scene(pane);
        defaultWindow.setScene(scene);
        defaultWindow.sizeToScene();
        defaultWindow.setTitle("Default Index Input");
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

        // Track class and type selection
        Label classLabel = new Label("Class of Track: ");
        ComboBox<String> classCombo = new ComboBox<>();
        classCombo.getItems().addAll("1", "2", "3", "4", "5");

        Label typeLabel = new Label("Type of Track: ");
        ComboBox<String> typeCombo = new ComboBox<>();
        typeCombo.getItems().addAll("Line (Straight)", "31-foot Chord", "62-foot Chord", 
                                    "31-foot Qualified Cant Chord", "62-foot Qualified Cant Chord");

        Label longitudinalLabel = new Label("Longitudinal Observation: ");
        TextField longitudinalInput = new TextField();
        configureInputField(longitudinalInput);

        Label alignmentLabel = new Label("Alignment Observation: ");
        TextField alignmentInput = new TextField();
        configureInputField(alignmentInput);

        Label gaugeLabel = new Label("Gauge Observation: ");
        TextField gaugeInput = new TextField();
        configureInputField(gaugeInput);

        Label WLLabel = new Label("WL: ");
        TextField WLInput = new TextField();
        configurePositiveInputField(WLInput);

        Label WALabel = new Label("WA: ");
        TextField WAInput = new TextField();
        configurePositiveInputField(WAInput);

        Label WGLabel = new Label("WG: ");
        TextField WGInput = new TextField();
        configurePositiveInputField(WGInput);

        Label errorMessage = new Label();
        errorMessage.setStyle("-fx-text-fill: red;");

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
        gridPane.add(WLLabel, 0, 6);
        gridPane.add(WLInput, 1, 6);
        gridPane.add(WALabel, 0, 7);
        gridPane.add(WAInput, 1, 7);
        gridPane.add(WGLabel, 0, 8);
        gridPane.add(WGInput, 1, 8);
        gridPane.add(errorMessage, 1, 9);

        pane.setCenter(gridPane);

        Button enterButton = new Button("Enter");
        enterButton.setOnAction(e -> {
            try {
                float l = Float.parseFloat(longitudinalInput.getText());
                float a = Float.parseFloat(alignmentInput.getText());
                float g = Float.parseFloat(gaugeInput.getText());
                float WL = Float.parseFloat(WLInput.getText());
                float WA = Float.parseFloat(WAInput.getText());
                float WG = Float.parseFloat(WGInput.getText());

                int trackClass = Integer.parseInt(classCombo.getValue());
                String trackType = typeCombo.getValue();

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
                        case 1 -> { Lmax = 0; Gmax = 1; Amax = 0; } // N/A for L and A
                        case 2 -> { Lmax = 0; Gmax = 0.875f; Amax = 0; } // N/A for L and A
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
                        case 1 -> { Lmax = 0; Gmax = 1; Amax = 0; } // N/A for L and A
                        case 2 -> { Lmax = 0; Gmax = 0.875f; Amax = 0; } // N/A for L and A
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

                l = Math.max(0, l - Lmax);
                a = Math.max(0, a - Amax);
                g = Math.max(0, g - Gmax);

                varTGIone(l, a, g, WL, WA, WG, Lmax, Amax, Gmax);
                VarOneWindow.close();
            } catch (NumberFormatException ex) {
                errorMessage.setText("Invalid input. Please enter numerical values.");
            }
        });

        BorderPane bottomPane = new BorderPane();
        bottomPane.setRight(enterButton);
        BorderPane.setMargin(enterButton, new Insets(10));
        pane.setBottom(bottomPane);

        Scene scene = new Scene(pane);
        VarOneWindow.setScene(scene);
        VarOneWindow.sizeToScene();
        VarOneWindow.setTitle("Variation 1 Observation Input");
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

        // Track class and type selection
        Label classLabel = new Label("Class of Track: ");
        ComboBox<String> classCombo = new ComboBox<>();
        classCombo.getItems().addAll("1", "2", "3", "4", "5");

        Label typeLabel = new Label("Type of Track: ");
        ComboBox<String> typeCombo = new ComboBox<>();
        typeCombo.getItems().addAll("Line (Straight)", "31-foot Chord", "62-foot Chord", 
                                    "31-foot Qualified Cant Chord", "62-foot Qualified Cant Chord");

        Label longitudinalLabel = new Label("Longitudinal Observation: ");
        TextField longitudinalInput = new TextField();
        configureInputField(longitudinalInput);

        Label alignmentLabel = new Label("Alignment Observation: ");
        TextField alignmentInput = new TextField();
        configureInputField(alignmentInput);

        Label gaugeLabel = new Label("Gauge Observation: ");
        TextField gaugeInput = new TextField();
        configureInputField(gaugeInput);

        Label errorMessage = new Label();
        errorMessage.setStyle("-fx-text-fill: red;");

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
        gridPane.add(errorMessage, 1, 6);

        pane.setCenter(gridPane);

        Button enterButton = new Button("Enter");
        enterButton.setOnAction(e -> {
            try {
                // Parse input values
                float l = Float.parseFloat(longitudinalInput.getText());
                float a = Float.parseFloat(alignmentInput.getText());
                float g = Float.parseFloat(gaugeInput.getText());

                int trackClass = Integer.parseInt(classCombo.getValue());
                String trackType = typeCombo.getValue();

                // Get thresholds based on class and type
                float Lmax = 0, Amax = 0, Gmax = 0;
                switch (trackType) {
                    case "Line (Straight)":
                        switch (trackClass) {
                            case 1 -> { Lmax = 3; Gmax = 1; Amax = 5; }
                            case 2 -> { Lmax = 2; Gmax = 0.875f; Amax = 3; }
                            case 3 -> { Lmax = 1.75f; Gmax = 0.875f; Amax = 1.75f; }
                            case 4 -> { Lmax = 1.5f; Gmax = 0.75f; Amax = 1.5f; }
                            case 5 -> { Lmax = 1; Gmax = 0.75f; Amax = 0.75f; }
                        }
                        break;
                    case "31-foot Chord":
                        switch (trackClass) {
                            case 1 -> { Lmax = 0; Gmax = 1; Amax = 0; } // N/A for L and A
                            case 2 -> { Lmax = 0; Gmax = 0.875f; Amax = 0; } // N/A for L and A
                            case 3 -> { Lmax = 1; Gmax = 0.875f; Amax = 1.25f; }
                            case 4 -> { Lmax = 1; Gmax = 0.75f; Amax = 1; }
                            case 5 -> { Lmax = 1; Gmax = 0.75f; Amax = 0.5f; }
                        }
                        break;
                    case "62-foot Chord":
                        switch (trackClass) {
                            case 1 -> { Lmax = 3; Gmax = 1; Amax = 5; }
                            case 2 -> { Lmax = 2.75f; Gmax = 0.875f; Amax = 3; }
                            case 3 -> { Lmax = 2.25f; Gmax = 0.875f; Amax = 1.75f; }
                            case 4 -> { Lmax = 2; Gmax = 0.75f; Amax = 1.5f; }
                            case 5 -> { Lmax = 1.25f; Gmax = 0.75f; Amax = 0.625f; }
                        }
                        break;
                    case "31-foot Qualified Cant Chord":
                        switch (trackClass) {
                            case 1 -> { Lmax = 0; Gmax = 1; Amax = 0; } // N/A for L and A
                            case 2 -> { Lmax = 0; Gmax = 0.875f; Amax = 0; } // N/A for L and A
                            case 3 -> { Lmax = 1; Gmax = 0.875f; Amax = 1.25f; }
                            case 4 -> { Lmax = 1; Gmax = 0.75f; Amax = 1; }
                            case 5 -> { Lmax = 1; Gmax = 0.75f; Amax = 0.5f; }
                        }
                        break;
                    case "62-foot Qualified Cant Chord":
                        switch (trackClass) {
                            case 1 -> { Lmax = 2.25f; Gmax = 1; Amax = 1.25f; }
                            case 2 -> { Lmax = 2.25f; Gmax = 0.875f; Amax = 1.25f; }
                            case 3 -> { Lmax = 1.75f; Gmax = 0.875f; Amax = 1.25f; }
                            case 4 -> { Lmax = 1.25f; Gmax = 0.75f; Amax = 0.875f; }
                            case 5 -> { Lmax = 1; Gmax = 0.75f; Amax = 0.625f; }
                        }
                        break;
                }

                // Adjust input values by thresholds
                l = Math.max(0, l - Lmax);
                a = Math.max(0, a - Amax);
                g = Math.max(0, g - Gmax);

                // Calculate user TGI
                varTGItwo(l, a, g, Lmax, Amax, Gmax);
                VarTwoWindow.close();
            } catch (NumberFormatException ex) {
                errorMessage.setText("Invalid input. Please enter numerical values.");
            }
        });

        BorderPane bottomPane = new BorderPane();
        bottomPane.setRight(enterButton);
        BorderPane.setMargin(enterButton, new Insets(10));
        pane.setBottom(bottomPane);

        Scene scene = new Scene(pane);
        VarTwoWindow.setScene(scene);
        VarTwoWindow.sizeToScene();
        VarTwoWindow.setTitle("Variation 2 Observation Input");
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

        Scene scene = new Scene(pane);
        NTQIWindow.setScene(scene);
        NTQIWindow.sizeToScene();
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

        Label classLabel = new Label("Track Class: ");
        ComboBox<String> classCombo = new ComboBox<>();
        classCombo.getItems().addAll("K0","K1", "K2", "K3", "K4", "K5");

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

        Button nextButton = new Button("Enter");

        nextButton.setOnAction(e -> {
            try {
            	String trackClass = classCombo.getValue(); 
                float[] limits = getTrackClassLimits(trackClass); 

                if (!selectedFileLabel.getText().equals("No file selected")) {
                    File excelFile = new File(selectedFileLabel.getText());

                    List<Float> HList = new ArrayList<>();
                    List<Float> SList = new ArrayList<>();
                    
                    try (FileInputStream fis = new FileInputStream(excelFile);
                         Workbook workbook = new XSSFWorkbook(fis)) {
                        Sheet sheet = workbook.getSheetAt(0);

                        for (Row row : sheet) {
                            if (row.getCell(0).getCellType() == CellType.NUMERIC && row.getCell(1).getCellType() == CellType.NUMERIC) {
                                float hValue = (float) row.getCell(0).getNumericCellValue(); // First column is H
                                float sValue = (float) row.getCell(1).getNumericCellValue(); // Second column is S
                                HList.add(hValue);
                                SList.add(sValue);
                            } else {
                                throw new IllegalArgumentException("Invalid format: Non-numeric values detected.");
                            }
                        }
                    } catch (IOException | IllegalArgumentException ex) {
                        showError("Error reading Excel file or invalid format. Ensure proper data. Click info for help.");
                    }

                    varTGIswedenQ(HList.size(), limits[0], limits[1], HList, SList);
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

        Scene scene = new Scene(pane);
        swedenQWindow.setScene(scene);
        swedenQWindow.sizeToScene();
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

        Scene scene = new Scene(pane);
        jCoeffWindow.setScene(scene);
        jCoeffWindow.sizeToScene();
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

        Scene scene = new Scene(pane);
        CNWindow.setScene(scene);
        CNWindow.sizeToScene();
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

        Scene scene = new Scene(pane);
        TGIWindow.setScene(scene);
        TGIWindow.sizeToScene();
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

        Scene scene = new Scene(pane);
        TGIVarWindow.setScene(scene);
        TGIVarWindow.sizeToScene();
        TGIVarWindow.setTitle("TGI Variation Input");
        TGIVarWindow.show();
    }
     
    // output methods, assumes that we have the components necessary for final computation, displays output
    
    private void varDefaultTGI(int instances, float Lmax, float Amax, float Gmax, List<float[]> longitudinalList, List<float[]> alignmentList, List<float[]> gaugeList) {
        List<Float> tgiValues = new ArrayList<>();
        int satisfactoryInstancesL = 0;
        int satisfactoryInstancesA = 0;
        int satisfactoryInstancesG = 0;
        int satisfactoryInstancesOverall = 0;

        StringBuilder exceedanceOutput = new StringBuilder();

        for (int i = 0; i < instances; i++) {
            float l = longitudinalList.get(i)[0];
            float a = alignmentList.get(i)[0];
            float g = gaugeList.get(i)[0];

            float exceedL = l - Lmax > 0 ? l - Lmax : 0;
            float exceedA = a - Amax > 0 ? a - Amax : 0;
            float exceedG = g - Gmax > 0 ? g - Gmax : 0;

            // Calculate the actual TGI
            float tgiOut = 100 - (((exceedL + exceedA + exceedG) / (Lmax + Amax + Gmax)) * 100);
            tgiOut = Math.round(tgiOut);
            tgiValues.add(tgiOut);

            if (l <= Lmax) satisfactoryInstancesL++;
            if (a <= Amax) satisfactoryInstancesA++;
            if (g <= Gmax) satisfactoryInstancesG++;
            if (l <= Lmax && a <= Amax && g <= Gmax) satisfactoryInstancesOverall++;

            exceedanceOutput.append(String.format("Instance %d L, A, G exceed: %.2f, %.2f, %.2f\n", i + 1, exceedL, exceedA, exceedG));
        }

        float KL = (float) satisfactoryInstancesL / instances;
        float KA = (float) satisfactoryInstancesA / instances;
        float KG = (float) satisfactoryInstancesG / instances;
        float KOverall = (float) satisfactoryInstancesOverall / instances;

        // Output stage
        resultStage = new Stage();
        BorderPane resultPane = new BorderPane();
        resultPane.setPadding(new Insets(10));

        VBox resultBox = new VBox(10);
        resultBox.setPadding(new Insets(10));
        resultBox.setAlignment(Pos.CENTER_LEFT);

        resultBox.getChildren().add(new Label("TGI Values: " + tgiValues.toString()));
        resultBox.getChildren().add(new Label(String.format("K (L): %.3f", KL)));
        resultBox.getChildren().add(new Label(String.format("K (A): %.3f", KA)));
        resultBox.getChildren().add(new Label(String.format("K (G): %.3f", KG)));
        resultBox.getChildren().add(new Label(String.format("K (Overall): %.3f", KOverall)));
        resultBox.getChildren().add(new Label("Exceedance Values:"));
        resultBox.getChildren().add(new Label(exceedanceOutput.toString()));

        Button saveButton = new Button("Save to Excel");
        saveButton.setOnAction(e -> {
            saveDefault(instances, longitudinalList, alignmentList, gaugeList, tgiValues, KL, KA, KG, KOverall, Lmax, Amax, Gmax);
            resultStage.close();
        });

        Button helpButton = new Button("?");
        helpButton.setStyle(
            "-fx-background-color: lightgray; " +
            "-fx-text-fill: black; " +
            "-fx-font-weight: bold; " +
            "-fx-background-radius: 15; " +
            "-fx-padding: 2; " +
            "-fx-min-width: 20px; " +
            "-fx-min-height: 20px; " +
            "-fx-max-width: 20px; " +
            "-fx-max-height: 20px;"
        );
        helpButton.setOnAction(e -> showInfoPopup());

        HBox buttonBox = new HBox(10);
        buttonBox.getChildren().addAll(saveButton, helpButton);

        resultBox.getChildren().add(buttonBox);

        ScrollPane scrollPane = new ScrollPane(resultBox);
        scrollPane.setFitToWidth(true);

        resultPane.setCenter(scrollPane);
        Scene resultScene = new Scene(resultPane);
        resultStage.setScene(resultScene);
        resultStage.sizeToScene();
        resultStage.setTitle("Default Index Results");
        resultStage.show();
    }
    
    private void varTGIone(float l, float a, float g, float WL, float WA, float WG, 
            float Lmax, float Amax, float Gmax) {
    			float userNum = ((WL * l) + (WA * a) + (WG * g));
    			float userFactor = userNum / 10; 
    			float userTGI = 100 - (userFactor * 100);
    			userTGI = Math.max(userTGI, 0); 

    			float thresholdNum = ((WL * Lmax) + (WA * Amax) + (WG * Gmax));
    			float thresholdFactor = thresholdNum / 10;
    			float thresholdTGI = 100 - (thresholdFactor * 100);
    			thresholdTGI = Math.max(thresholdTGI, 0); 

    			String[] assignments = { "Very Poor", "Poor", "Fair", "Good", "Excellent" };
    			String condition = assignments[(int) Math.min(Math.max(userTGI / 20, 0), 4)];
    			String[] recommendations = {
    					"Immediate shutdown or major repairs required",
    					"Urgent repairs necessary; restrict speeds",
    					"Immediate corrective actions planned",
    					"Schedule preventive maintenance",
    					"Routine monitoring, no immediate action required"
    			};
    			String recommendedCourse = recommendations[(int) Math.min(userTGI / 20, 4)];

    			Stage resultStage = new Stage();
    			BorderPane resultPane = new BorderPane();
    			resultPane.setPadding(new Insets(10));

    			VBox resultBox = new VBox(10);
    			resultBox.setPadding(new Insets(10));
    			resultBox.setAlignment(Pos.CENTER_LEFT);

    			resultBox.getChildren().add(new Label("User TGI Output: " + String.format("%.2f", userTGI)));
    			resultBox.getChildren().add(new Label("User Condition: " + condition));
    			resultBox.getChildren().add(new Label("Recommended Course of Action: " + recommendedCourse));
    			resultBox.getChildren().add(new Label("Threshold TGI: " + String.format("%.2f", thresholdTGI)));

    			Button saveButton = new Button("Save to Excel");
    	        saveButton.setOnAction(e -> {
    	            // save method here
    	            resultStage.close();  
    	        });

    	        Button helpButton = new Button("?");
    	        helpButton.setStyle(
    	            "-fx-background-color: lightgray; " +   
    	            "-fx-text-fill: black; " +              
    	            "-fx-font-weight: bold; " +             
    	            "-fx-background-radius: 15; " +         
    	            "-fx-padding: 2; " +                    
    	            "-fx-min-width: 20px; " +               
    	            "-fx-min-height: 20px; " +              
    	            "-fx-max-width: 20px; " +               
    	            "-fx-max-height: 20px;"                 
    	        );
    	        helpButton.setOnAction(e -> showInfoPopup());  // Placeholder for future functionality

    	        HBox buttonBox = new HBox(10); 
    	        buttonBox.getChildren().addAll(saveButton, helpButton);

    	        resultBox.getChildren().add(buttonBox);
    	        
    	        ScrollPane scrollPane = new ScrollPane(resultBox);
    	        scrollPane.setFitToWidth(true);

    	        resultPane.setCenter(scrollPane);
    	        Scene resultScene = new Scene(resultPane);
    			resultStage.setScene(resultScene);
    			resultStage.sizeToScene();
    			resultStage.setTitle("Variation 1 Results");
    			resultStage.show();
    }
    
    private void varTGItwo(float l, float a, float g, float Lmax, float Amax, float Gmax) {
        // Invert TGI calculation: 100 is the maximum score, with no deviations.
        float tgiOut = 100 - (float) Math.sqrt(l * l + a * a + g * g);
        
        // Clamp TGI output so it doesn't go below 0.
        tgiOut = Math.max(tgiOut, 0);

        String condition;
        String recommendedCourse;

        // Determine condition and recommendation based on TGI value
        if (tgiOut >= 80) {
            condition = "Good Quality";
            recommendedCourse = "Routine Monitoring";
        } else if (tgiOut >= 60) {
            condition = "Fair Quality";
            recommendedCourse = "Planned maintenance needed";
        } else if (tgiOut >= 40) {
            condition = "Poor Quality";
            recommendedCourse = "Immediate corrective actions required";
        } else if (tgiOut >= 20) {
            condition = "Very Poor Quality";
            recommendedCourse = "Urgent repairs needed";
        } else {
            condition = "Very Poor";
            recommendedCourse = "Urgent repairs needed";
        }

        // Calculate TGI for threshold values (this provides a comparison against ideal threshold values)
        float thresholdTGI = 100 - (float) Math.sqrt(Lmax * Lmax + Amax * Amax + Gmax * Gmax);
        thresholdTGI = Math.max(thresholdTGI, 0);  // Ensure it doesn't go below 0.

        // Output stage
        resultStage = new Stage();
        BorderPane resultPane = new BorderPane();
        resultPane.setPadding(new Insets(10));

        VBox resultBox = new VBox(10);
        resultBox.setPadding(new Insets(10));
        resultBox.setAlignment(Pos.CENTER_LEFT);

        // Show the user's TGI and how it compares to the threshold TGI
        resultBox.getChildren().add(new Label("TGI Output: " + tgiOut));
        resultBox.getChildren().add(new Label("Condition: " + condition));
        resultBox.getChildren().add(new Label("Recommended Course of Action: " + recommendedCourse));
        resultBox.getChildren().add(new Label("TGI (Threshold): " + thresholdTGI));

        Button saveButton = new Button("Save to Excel");
        saveButton.setOnAction(e -> {
            // save method here
            resultStage.close();  
        });

        Button helpButton = new Button("?");
        helpButton.setStyle(
            "-fx-background-color: lightgray; " +   
            "-fx-text-fill: black; " +              
            "-fx-font-weight: bold; " +             
            "-fx-background-radius: 15; " +         
            "-fx-padding: 2; " +                    
            "-fx-min-width: 20px; " +               
            "-fx-min-height: 20px; " +              
            "-fx-max-width: 20px; " +               
            "-fx-max-height: 20px;"                 
        );
        helpButton.setOnAction(e -> showInfoPopup());

        HBox buttonBox = new HBox(10); 
        buttonBox.getChildren().addAll(saveButton, helpButton);

        resultBox.getChildren().add(buttonBox);
        
        ScrollPane scrollPane = new ScrollPane(resultBox);
        scrollPane.setFitToWidth(true);

        resultPane.setCenter(scrollPane);
        
        Scene resultScene = new Scene(resultPane);
        resultStage.setScene(resultScene);
        resultStage.sizeToScene();
        resultStage.setTitle("Variation 2 Results");
        resultStage.show();
    }
    
    private void varTGIntqi(List<Double> stdDevs, double percentile80) {
        resultStage = new Stage();
        BorderPane resultPane = new BorderPane();
        resultPane.setPadding(new Insets(10));

        VBox resultBox = new VBox(10);
        resultBox.setPadding(new Insets(10));
        resultBox.setAlignment(Pos.CENTER_LEFT);

        for (int i = 0; i < stdDevs.size(); i++) {
            double stddev = stdDevs.get(i);
            double factor = stddev / percentile80;
            float tgiOut = (float) (10 * Math.pow(0.675, factor));

            resultBox.getChildren().add(new Label("Segment " + (i + 1) + " Output: " + tgiOut));
        }

        Button saveButton = new Button("Save to Excel");
        saveButton.setOnAction(e -> {
            // save method here
            resultStage.close();  
        });

        Button helpButton = new Button("?");
        helpButton.setStyle(
            "-fx-background-color: lightgray; " +   
            "-fx-text-fill: black; " +              
            "-fx-font-weight: bold; " +             
            "-fx-background-radius: 15; " +         
            "-fx-padding: 2; " +                    
            "-fx-min-width: 20px; " +               
            "-fx-min-height: 20px; " +              
            "-fx-max-width: 20px; " +               
            "-fx-max-height: 20px;"                 
        );
        helpButton.setOnAction(e -> showInfoPopup());  // Placeholder for future functionality

        HBox buttonBox = new HBox(10); 
        buttonBox.getChildren().addAll(saveButton, helpButton);

        resultBox.getChildren().add(buttonBox);
        
        ScrollPane scrollPane = new ScrollPane(resultBox);
        scrollPane.setFitToWidth(true);

        resultPane.setCenter(scrollPane);

        Scene resultScene = new Scene(resultPane);
        resultStage.setScene(resultScene);
        resultStage.sizeToScene();
        resultStage.setTitle("NTQI Results");
        resultStage.show();
    }
   
    private void varTGIswedenQ(int instances, float Hlim, float Slim, List<Float> HList, List<Float> SList) {
        List<Float> tgiValues = new ArrayList<>();
        int satisfactoryInstancesH = 0;
        int satisfactoryInstancesS = 0;
        int satisfactoryInstancesOverall = 0;

        for (int i = 0; i < instances; i++) {
            float sigmaH = HList.get(i);
            float sigmaS = SList.get(i);

            float normalizedH = sigmaH / Hlim;
            float normalizedS = 2 * (sigmaS / Slim);

            float totalDeviation = normalizedH + normalizedS;
            int roundedDeviation = (int) Math.ceil(totalDeviation); // Ceiling of the total deviation
            float tgiOut = 150 - (100.0f / 3.0f) * roundedDeviation; // Sweden QI formula

            tgiValues.add(tgiOut);

            if (sigmaH <= Hlim) satisfactoryInstancesH++;
            if (sigmaS <= Slim) satisfactoryInstancesS++;
            if (sigmaH <= Hlim && sigmaS <= Slim) satisfactoryInstancesOverall++;
        }

        // Calculate the K values
        float KH = (float) satisfactoryInstancesH / instances;
        float KS = (float) satisfactoryInstancesS / instances;
        float KOverall = (float) satisfactoryInstancesOverall / instances;

        // Display the results
        resultStage = new Stage();
        BorderPane resultPane = new BorderPane();
        resultPane.setPadding(new Insets(10));

        VBox resultBox = new VBox(10);
        resultBox.setPadding(new Insets(10));
        resultBox.setAlignment(Pos.CENTER_LEFT);

        resultBox.getChildren().add(new Label("QI Values: " + tgiValues.toString()));
        resultBox.getChildren().add(new Label(String.format("K (Hlim): %.3f", KH)));
        resultBox.getChildren().add(new Label(String.format("K (Slim): %.3f", KS)));
        resultBox.getChildren().add(new Label(String.format("K (Overall): %.3f", KOverall)));

        Button saveButton = new Button("Save to Excel");
        saveButton.setOnAction(e -> {
            // save method here
            resultStage.close();  
        });

        Button helpButton = new Button("?");
        helpButton.setStyle(
            "-fx-background-color: lightgray; " +   
            "-fx-text-fill: black; " +              
            "-fx-font-weight: bold; " +             
            "-fx-background-radius: 15; " +         
            "-fx-padding: 2; " +                    
            "-fx-min-width: 20px; " +               
            "-fx-min-height: 20px; " +              
            "-fx-max-width: 20px; " +               
            "-fx-max-height: 20px;"                 
        );
        helpButton.setOnAction(e -> showInfoPopup());  // Placeholder for future functionality

        HBox buttonBox = new HBox(10); 
        buttonBox.getChildren().addAll(saveButton, helpButton);

        resultBox.getChildren().add(buttonBox);
        
        ScrollPane scrollPane = new ScrollPane(resultBox);
        scrollPane.setFitToWidth(true);

        resultPane.setCenter(scrollPane);

        Scene resultScene = new Scene(resultPane);
        resultStage.setScene(resultScene);
        resultStage.sizeToScene();
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

        Button saveButton = new Button("Save to Excel");
        saveButton.setOnAction(e -> {
            // save method here
            resultStage.close();  
        });

        Button helpButton = new Button("?");
        helpButton.setStyle(
            "-fx-background-color: lightgray; " +   
            "-fx-text-fill: black; " +              
            "-fx-font-weight: bold; " +             
            "-fx-background-radius: 15; " +         
            "-fx-padding: 2; " +                    
            "-fx-min-width: 20px; " +               
            "-fx-min-height: 20px; " +              
            "-fx-max-width: 20px; " +               
            "-fx-max-height: 20px;"                 
        );
        helpButton.setOnAction(e -> showInfoPopup());  // Placeholder for future functionality

        HBox buttonBox = new HBox(10); 
        buttonBox.getChildren().addAll(saveButton, helpButton);

        resultBox.getChildren().add(buttonBox);
        
        ScrollPane scrollPane = new ScrollPane(resultBox);
        scrollPane.setFitToWidth(true);

        resultPane.setCenter(scrollPane);

        Scene resultScene = new Scene(resultPane);
        resultStage.sizeToScene();
        resultStage.setScene(resultScene);
        resultStage.setTitle("J Coefficient Results");
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
            resultBox.getChildren().add(new Label(categories[i] + " Score: " + String.format("%.2f", tgiValues[i])));
        }
        resultBox.getChildren().add(new Label("Overall Score: " + String.format("%.2f", averageTQI)));

        Button saveButton = new Button("Save to Excel");
        saveButton.setOnAction(e -> {
            // save method here
            resultStage.close();  
        });

        Button helpButton = new Button("?");
        helpButton.setStyle(
            "-fx-background-color: lightgray; " +   
            "-fx-text-fill: black; " +              
            "-fx-font-weight: bold; " +             
            "-fx-background-radius: 15; " +         
            "-fx-padding: 2; " +                    
            "-fx-min-width: 20px; " +               
            "-fx-min-height: 20px; " +              
            "-fx-max-width: 20px; " +               
            "-fx-max-height: 20px;"                 
        );
        helpButton.setOnAction(e -> showInfoPopup());  // Placeholder for future functionality

        HBox buttonBox = new HBox(10); 
        buttonBox.getChildren().addAll(saveButton, helpButton);

        resultBox.getChildren().add(buttonBox);
        
        ScrollPane scrollPane = new ScrollPane(resultBox);
        scrollPane.setFitToWidth(true);

        resultPane.setCenter(scrollPane);

        Scene resultScene = new Scene(resultPane);
        resultStage.sizeToScene();
        resultStage.setScene(resultScene);
        resultStage.setTitle("CN Results");
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
        ui = Math.min(ui, 100);
        float ai = (float)(100 * (Math.exp(aiFac)));
        ai = Math.min(ai, 100);
        float ti = (float)(100 * (Math.exp(tiFac)));
        ti = Math.min(ti, 100);
        float gi = (float)(100 * (Math.exp(giFac)));
        gi = Math.min(gi, 100);
        
        float tgiNum = 2 * ui + ti + gi + 6 * ai;
        float TGI = tgiNum / 10;

        resultBox.getChildren().add(new Label("UI: " + String.format("%.2f", ui)));
        resultBox.getChildren().add(new Label("AI: " + String.format("%.2f", ai)));
        resultBox.getChildren().add(new Label("TI: " + String.format("%.2f", ti)));
        resultBox.getChildren().add(new Label("GI: " + String.format("%.2f", gi)));
        resultBox.getChildren().add(new Label("Overall TGI: " + String.format("%.2f", TGI)));

        Button saveButton = new Button("Save to Excel");
        saveButton.setOnAction(e -> {
            // save method here
            resultStage.close();  
        });

        Button helpButton = new Button("?");
        helpButton.setStyle(
            "-fx-background-color: lightgray; " +   
            "-fx-text-fill: black; " +              
            "-fx-font-weight: bold; " +             
            "-fx-background-radius: 15; " +         
            "-fx-padding: 2; " +                    
            "-fx-min-width: 20px; " +               
            "-fx-min-height: 20px; " +              
            "-fx-max-width: 20px; " +               
            "-fx-max-height: 20px;"                 
        );
        helpButton.setOnAction(e -> showInfoPopup());  // Placeholder for future functionality

        HBox buttonBox = new HBox(10); 
        buttonBox.getChildren().addAll(saveButton, helpButton);

        resultBox.getChildren().add(buttonBox);
        
        ScrollPane scrollPane = new ScrollPane(resultBox);
        scrollPane.setFitToWidth(true);

        resultPane.setCenter(scrollPane);

        Scene resultScene = new Scene(resultPane);
        resultStage.setScene(resultScene);
        resultStage.sizeToScene();
        resultStage.setTitle("TGI Results");
        resultStage.show();
    }
    
    private void varTGIvar(List<Float[]> instances, float Lmax, float Amax, float Gmax) {
        int satisfactoryCountL = 0;
        int satisfactoryCountA = 0;
        int satisfactoryCountG = 0;
        int satisfactoryCountOverall = 0;  
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

            if (lexceed == 0) satisfactoryCountL++;
            if (aexceed == 0) satisfactoryCountA++;
            if (gexceed == 0) satisfactoryCountG++;

            if (lexceed == 0 && aexceed == 0 && gexceed == 0) satisfactoryCountOverall++;

            results.add("Instance " + (i + 1) + ": TGI = " + String.format("%.2f", tgi) 
                        + ", Classification = " + classification 
                        + ", L exceed: " + String.format("%.2f", lexceed) 
                        + ", A exceed: " + String.format("%.2f", aexceed) 
                        + ", G exceed: " + String.format("%.2f", gexceed));
        }

        float KL = (float) satisfactoryCountL / instances.size();
        float KA = (float) satisfactoryCountA / instances.size();
        float KG = (float) satisfactoryCountG / instances.size();
        float KOverall = (float) satisfactoryCountOverall / instances.size();

        resultStage = new Stage();
        BorderPane resultPane = new BorderPane();
        resultPane.setPadding(new Insets(10));

        VBox resultBox = new VBox(10);
        resultBox.setPadding(new Insets(10));
        resultBox.setAlignment(Pos.CENTER_LEFT);

        for (String result : results) {
            resultBox.getChildren().add(new Label(result));
        }

        resultBox.getChildren().add(new Label(String.format("K (L): %.3f", KL)));
        resultBox.getChildren().add(new Label(String.format("K (A): %.3f", KA)));
        resultBox.getChildren().add(new Label(String.format("K (G): %.3f", KG)));
        resultBox.getChildren().add(new Label(String.format("K (Overall): %.3f", KOverall)));

        Button saveButton = new Button("Save to Excel");
        saveButton.setOnAction(e -> {
            // save method here
            resultStage.close();  
        });

        Button helpButton = new Button("?");
        helpButton.setStyle(
            "-fx-background-color: lightgray; " +   
            "-fx-text-fill: black; " +              
            "-fx-font-weight: bold; " +             
            "-fx-background-radius: 15; " +         
            "-fx-padding: 2; " +                    
            "-fx-min-width: 20px; " +               
            "-fx-min-height: 20px; " +              
            "-fx-max-width: 20px; " +               
            "-fx-max-height: 20px;"                 
        );
        helpButton.setOnAction(e -> showInfoPopup());  // Placeholder for future functionality

        HBox buttonBox = new HBox(10); 
        buttonBox.getChildren().addAll(saveButton, helpButton);

        resultBox.getChildren().add(buttonBox);
        
        ScrollPane scrollPane = new ScrollPane(resultBox);
        scrollPane.setFitToWidth(true);

        resultPane.setCenter(scrollPane);

        Scene resultScene = new Scene(resultPane);
        resultStage.setScene(resultScene);
        resultStage.sizeToScene();
        resultStage.setTitle("Variation 3 Results");
        resultStage.show();
    }
    
    private void saveDefault(int instances, List<float[]> longitudinalList, List<float[]> alignmentList, List<float[]> gaugeList,
            List<Float> tgiValues, float KL, float KA, float KG, float KOverall, float Lmax, float Amax, float Gmax) {

    // Retrieve the path from file_path.txt
    String filePath = getFilePath();
    File excelFile = new File(filePath + "/Default.xlsx");

    try (FileInputStream fis = new FileInputStream(excelFile);
        Workbook workbook = new XSSFWorkbook(fis)) {

        Sheet sheet = workbook.getSheetAt(0);
        int nextRow = getNextAvailableRow(sheet);  // Find the next empty row

        // **Step 1**: Always write headers after finding the next empty row and moving down by one row
        nextRow++;  // Move one row down to leave a blank row above the headers
        Row headerRow = sheet.createRow(nextRow++);
        headerRow.createCell(0).setCellValue("Run #");
        headerRow.createCell(1).setCellValue("Instance #");
        headerRow.createCell(2).setCellValue("L (Input)");
        headerRow.createCell(3).setCellValue("A (Input)");
        headerRow.createCell(4).setCellValue("G (Input)");
        headerRow.createCell(5).setCellValue("L Threshold");
        headerRow.createCell(6).setCellValue("A Threshold");
        headerRow.createCell(7).setCellValue("G Threshold");
        headerRow.createCell(8).setCellValue("L Exceed");
        headerRow.createCell(9).setCellValue("A Exceed");
        headerRow.createCell(10).setCellValue("G Exceed");
        headerRow.createCell(11).setCellValue("TGI (Instance)");

        // **Step 2**: Write instance data
        int runNumber = getNextRunNumber(sheet);  // Calculate the run number
        for (int i = 0; i < instances; i++) {
            Row row = sheet.createRow(nextRow++);
            row.createCell(0).setCellValue(runNumber);  // Run number
            row.createCell(1).setCellValue(i + 1);  // Instance #
            row.createCell(2).setCellValue(longitudinalList.get(i)[0]); // L (Input)
            row.createCell(3).setCellValue(alignmentList.get(i)[0]);    // A (Input)
            row.createCell(4).setCellValue(gaugeList.get(i)[0]);        // G (Input)
            row.createCell(5).setCellValue(Lmax);  // L Threshold
            row.createCell(6).setCellValue(Amax);  // A Threshold
            row.createCell(7).setCellValue(Gmax);  // G Threshold
            row.createCell(8).setCellValue(Math.max(0, longitudinalList.get(i)[0] - Lmax)); // L Exceed
            row.createCell(9).setCellValue(Math.max(0, alignmentList.get(i)[0] - Amax));    // A Exceed
            row.createCell(10).setCellValue(Math.max(0, gaugeList.get(i)[0] - Gmax));       // G Exceed
            row.createCell(11).setCellValue(tgiValues.get(i));           // TGI (Instance)
        }

        // **Step 3**: Write run-level summary with the current date
        Row summaryRow = sheet.createRow(nextRow++);
        SimpleDateFormat dateFormat = new SimpleDateFormat("MM/dd/yyyy");
        String currentDate = dateFormat.format(new Date());

        summaryRow.createCell(0).setCellValue("Run " + runNumber + " Summary:");
        summaryRow.createCell(1).setCellValue("K (L): " + KL);
        summaryRow.createCell(2).setCellValue("K (A): " + KA);
        summaryRow.createCell(3).setCellValue("K (G): " + KG);
        summaryRow.createCell(4).setCellValue("K (Overall): " + KOverall);
        summaryRow.createCell(5).setCellValue("Average TGI: " + calculateAverageTGI(tgiValues));
        summaryRow.createCell(6).setCellValue("Date: " + currentDate);  // Add the date

        // **Step 4**: Write the workbook back to the file
        try (FileOutputStream fos = new FileOutputStream(excelFile)) {
            workbook.write(fos);
        }

    } catch (IOException e) {
        e.printStackTrace();
        showError("Error writing to Excel file.");
    }
}

    
    private float calculateAverageTGI(List<Float> tgiValues) {
        float totalTGI = 0;
        for (float tgi : tgiValues) {
            totalTGI += tgi;
        }
        return totalTGI / tgiValues.size();
    }
    
    private int getNextRunNumber(Sheet sheet) {
        int highestRunNumber = 0;

        // Loop through all rows to find the highest run number
        for (int i = 0; i <= sheet.getLastRowNum(); i++) {
            Row row = sheet.getRow(i);
            if (row != null && row.getCell(0) != null && row.getCell(0).getCellType() == CellType.NUMERIC) {
                int runNumber = (int) row.getCell(0).getNumericCellValue();
                if (runNumber > highestRunNumber) {
                    highestRunNumber = runNumber;
                }
            }
        }

        return highestRunNumber + 1;  // Return the next run number
    }

    private int getNextAvailableRow(Sheet sheet) {
        int rowNum = sheet.getLastRowNum();
        while (rowNum >= 0 && (sheet.getRow(rowNum) == null || sheet.getRow(rowNum).getCell(0) == null || 
                               sheet.getRow(rowNum).getCell(0).getStringCellValue().trim().isEmpty())) {
            rowNum--;
        }
        return rowNum + 1;  // Return the next empty row
    }
}