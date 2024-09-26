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
    
    public static float calc80(float[] values) { //helper method for 80th percentile 
        if (values == null || values.length == 0) {
            throw new IllegalArgumentException("Array must contain at least one element.");
        }
        
        Arrays.sort(values);

        int index = (int) Math.ceil(0.8 * values.length) - 1; 

        return values[index];
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

        for (int i = 1; i <= 10; i++) {
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
            else if ("Variant 6".equals(userChoice)) {
            	openVarSixWindow();
            }
            else if ("Variant 7".equals(userChoice)) {
            	openVarSevenWindow();
            }
            else if ("Variant 8".equals(userChoice)) {
            	openVarEightWindow();
            }
            else if ("Variant 9".equals(userChoice)) {
            	openVarNineWindow();
            }
            else if ("Variant 10".equals(userChoice)) {
            	openVarTenWindow();
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
    
    private void openVarFourWindow() { // this method determines the sum of l THAT, using var 5, is all COMPLETELY UNDER THRESHOLD, against total
    	// length. this means that this method needs to be altered significantly, but the base process below is correct. We just need the logic
    	// before these simple calculations are done according to var 5, which will depend on class, THIS WILL NEED TO BE MERGED 
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
                varTGIfive(instances, trackLength, limits[0], limits[1], HLeftList, HRightList, crossLevelsList, gaugesList, horizontalDeviationsList);
                varFiveWindow.close();

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
        varFiveWindow.setScene(scene);
        varFiveWindow.setTitle("Variation 5 Input");
        varFiveWindow.show();
    }
    
    private void collectInstanceData(List<float[]> HLeftList, List<float[]> HRightList, List<float[]> crossLevelsList, List<float[]> gaugesList, List<float[]> horizontalDeviationsList, int instanceNumber) {
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
        switch (trackClass) {
            case 1: return new float[] {3, 3};
            case 2: return new float[] {2, 2};
            case 3: return new float[] {1.75f, 1.75f};
            case 4: return new float[] {1.5f, 1.5f};
            case 5: return new float[] {1, 1};
            default: throw new IllegalArgumentException("Invalid Track Class");
        }
    }
    
    private void openVarSixWindow() {
        Stage varSixWindow = new Stage();
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
                varTGIsix(SDz, SDy, SDw, SDe);
                varSixWindow.close();
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
        varSixWindow.setScene(scene);
        varSixWindow.setTitle("Variation 6 Deviation Input");
        varSixWindow.show();
    }
    
    private void openVarSevenWindow() { // this method needs read and subsequent stdv calculation for each i, so each n is 
    	// multiple sections of track and will have many associated values, but only one stdv value, which is the part being 
    	// modeled here (we jump in at having all stdv values, when they should be calculated from multiple read in lists)
        Stage VarSevenWindow = new Stage();
        BorderPane pane = new BorderPane();
        pane.setPadding(new Insets(10));

        GridPane gridPane = new GridPane();
        gridPane.setPadding(new Insets(10));
        gridPane.setHgap(10);
        gridPane.setVgap(10);
        gridPane.setAlignment(Pos.CENTER_LEFT);

        Label notice = new Label("All measurements in mm");
        notice.setStyle("-fx-font-weight: bold;");

        Label stdvLabel = new Label("Standard Deviations (Comma Separated list): ");
        TextField stdvInput = new TextField();
        configureInputField(stdvInput);

        gridPane.add(notice, 0, 0);
        gridPane.add(stdvLabel, 0, 1);
        gridPane.add(stdvInput, 1, 1);
        
        pane.setCenter(gridPane);
        
        Button enterButton = new Button("Enter");
        enterButton.setOnAction(e -> {
            try {
                float[] Z = parseInputToFloatArray(stdvInput.getText());
                float[] unsorted = Arrays.copyOf (Z, Z.length);
                float eighty = calc80(Z);
  
                varTGIseven(unsorted, eighty);
                VarSevenWindow.close();
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

        Scene scene = new Scene(pane, 400, 300);
        VarSevenWindow.setScene(scene);
        VarSevenWindow.setTitle("Variation 2 Deviation Input");
        VarSevenWindow.show();
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
    
    private void openVarEightWindow() { // this allows user input for testing. The list values need to be read in via excel.
        Stage varEightWindow = new Stage();
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

                varTGIeight(stdvGauge, stdvCross, stdvLeftA, stdvRightA, stdvLeftS, stdvRightS);
                varEightWindow.close();
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
        varEightWindow.setScene(scene);
        varEightWindow.setTitle("Variation 8 Deviation Input");
        varEightWindow.show();
    }
    
    private void openVarNineWindow() {
        Stage varNineWindow = new Stage();
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

        // Add elements to the grid pane
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
                // Set static SDnew variables for each category
                float SDnewLong = 2.5f;
                float SDnewAlign = 1.5f;
                float SDnewGauge = 1.0f;
                float SDnewTwist = 1.75f;

                // Determine SDmain based on user selection
                float SDmainLong = above105.isSelected() ? 6.2f : 7.2f;
                float SDmainAlign = 3.0f; // Same for both speed settings
                float SDmainGauge = 3.6f; // Same for both speed settings
                float SDmainTwist = above105.isSelected() ? 3.8f : 4.2f;

                // Read inputs into float arrays
                float[] longitudinalLevel = parseInputToFloatArray(hLeftInput.getText());
                float[] alignment = parseInputToFloatArray(hRightInput.getText());
                float[] gauge = parseInputToFloatArray(crossLevelsInput.getText());
                float[] twist = parseInputToFloatArray(gaugesInput.getText());
                
                float SDu = (float)stdv(longitudinalLevel);
                float SDa = (float)stdv(alignment);
                float SDg = (float)stdv(gauge);
                float SDt = (float)stdv(twist);

                // Call the next method with the necessary parameters
                varTGInine(SDu, SDa, SDt, SDg, SDnewLong, SDnewAlign, SDnewTwist, SDnewGauge, SDmainLong, SDmainAlign, SDmainTwist,SDmainGauge);
                varNineWindow.close();
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
        varNineWindow.setScene(scene);
        varNineWindow.setTitle("Variation 9 Input");
        varNineWindow.show();
    }
    
    private void openVarTenWindow() {
        Stage varTenWindow = new Stage();
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

            // Call varTGIseven() with correct parameters
            varTGIten(l, a, g, Lmax, Amax, Gmax);

            varTenWindow.close();
        });

        BorderPane bottomPane = new BorderPane();
        bottomPane.setRight(enterButton);
        BorderPane.setMargin(enterButton, new Insets(10));
        pane.setBottom(bottomPane);

        Scene scene = new Scene(pane, 400, 300);
        varTenWindow.setScene(scene);
        varTenWindow.setTitle("Track Geometry Index Input");
        varTenWindow.show();
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
    
    private void varTGIfive(int instances, float trackLength, float Hlim, float Slim, List<float[]> HLeftList, List<float[]> HRightList, List<float[]> crossLevelsList, List<float[]> gaugesList, List<float[]> horizontalDeviationsList) {
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

        // Output Results
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
    
    private void varTGIsix (float SDz, float SDy, float SDw, float SDe) {
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
    
    private void varTGIseven(float[] unsorted, float eighty) {
        resultStage = new Stage();
        BorderPane resultPane = new BorderPane();
        resultPane.setPadding(new Insets(10));

        VBox resultBox = new VBox(10);
        resultBox.setPadding(new Insets(10));
        resultBox.setAlignment(Pos.CENTER_LEFT);

        StringBuilder exceedingIndices = new StringBuilder();
        StringBuilder exceedingValues = new StringBuilder();

        for (int i = 0; i < unsorted.length; i++) {
            double N = (10 * 0.675 * unsorted[i]) / eighty;
            System.out.println("Run is: "+ unsorted [i] + " , " + eighty + " , " + N);

            if (N >= 6.75) {
                if (exceedingIndices.length() > 0) {
                    exceedingIndices.append(", ");
                    exceedingValues.append(", ");
                }
                exceedingIndices.append(i + 1);
                exceedingValues.append(unsorted[i]);
            }
        }

        String indicesMessage = exceedingIndices.length() > 0 ? 
            "Exceeding Rail Sections: " + exceedingIndices.toString() : 
            "No sections exceeding the threshold.";
        
        String valuesMessage = exceedingValues.length() > 0 ? 
            "Exceeding Rail Section values: " + exceedingValues.toString() : 
            "No values exceeding the threshold.";

        resultBox.getChildren().add(new Label(indicesMessage));
        resultBox.getChildren().add(new Label(valuesMessage));

        resultPane.setCenter(resultBox);

        Scene resultScene = new Scene(resultPane, 400, 250);
        resultStage.setScene(resultScene);
        resultStage.setTitle("Results");
        resultStage.show();
    }
    
    private void varTGIeight(float facOne, float facTwo, float facThree, float facFour, float facFive, float facSix) {
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
    
    private void varTGInine(float SDu, float SDa, float SDt, float SDg, float SDuN, float SDaN, float SDtN, float SDgN,
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
    
    private void varTGIten(float L, float A, float G, float Lmax, float Amax, float Gmax) {
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