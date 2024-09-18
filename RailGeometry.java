import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
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
        configureBoundedInputField(WLInput); // New method to enforce bounds

        Label WALabel = new Label("WA: ");
        TextField WAInput = new TextField();
        configurePositiveInputField(WAInput);
        configureBoundedInputField(WAInput); // New method to enforce bounds

        Label WGLabel = new Label("WG: ");
        TextField WGInput = new TextField();
        configurePositiveInputField(WGInput);
        configureBoundedInputField(WGInput); // New method to enforce bounds

        // Navigation between input fields
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
        VarTwoWindow.setTitle("Default Deviation Input");
        VarTwoWindow.show();
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
}