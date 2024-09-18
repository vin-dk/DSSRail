import javafx.application.Application;
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

public class MenuSelectionGUI extends Application {

    public float tgiOut = 0;
    public String condition = "";
    public String recommendedCourse = "";
    private Stage resultStage;  

    String[] assignments = {"Very Poor", "Poor", "Fair", "Good", "Excellent"};
    String[] recommendation = {"Immediate shutdown or major repairs required", "Urgent repairs necessary; restrict speeds",
            "Immediate corrective actions planned", "Schedule preventive maintenance", "Routine monitoring, no immediate action required"};

    private String userChoice = "None";  

    public void defaultTGI(float l, float a, float g, float allowanceL, float allowanceA, float allowanceG  ) {
        float exceedL = l - allowanceL > 0 ? l - allowanceL : 0;
        float exceedA = a - allowanceA > 0 ? a - allowanceA : 0;
        float exceedG = g - allowanceG > 0 ? g - allowanceG : 0;

        tgiOut = 100 - ((l + a + g) / (allowanceL + allowanceA + allowanceG)) * 100;

        if (tgiOut < 20) {
            condition = assignments[0];
            recommendedCourse = recommendation[0];
        } else if (tgiOut >= 20 && tgiOut < 40) {
            condition = assignments[1];
            recommendedCourse = recommendation[1];
        } else if (tgiOut >= 40 && tgiOut < 60) {
            condition = assignments[2];
            recommendedCourse = recommendation[2];
        } else if (tgiOut >= 60 && tgiOut < 80) {
            condition = assignments[3];
            recommendedCourse = recommendation[3];
        } else if (tgiOut >= 80 && tgiOut <= 100) {
            condition = assignments[4];
            recommendedCourse = recommendation[4];
        } else {
            System.out.print("System error");
        }

        openResultWindow(exceedL, exceedA, exceedG);  // Show exceedances in the result window
    }
    
    public void varTGIone (float l, float a, float g, float WL, float WA, float WG) {
    	tgiOut = 100 - ((WL*l+WA*a+WG*g)/1) * 100;
    	if (tgiOut >= 0 && tgiOut < 20) {
            condition = assignments[0];
            recommendedCourse = recommendation[0];
        } else if (tgiOut >= 20 && tgiOut < 40) {
            condition = assignments[1];
            recommendedCourse = recommendation[1];
        } else if (tgiOut >= 40 && tgiOut < 60) {
            condition = assignments[2];
            recommendedCourse = recommendation[2];
        } else if (tgiOut >= 60 && tgiOut < 80) {
            condition = assignments[3];
            recommendedCourse = recommendation[3];
        } else if (tgiOut >= 80 && tgiOut <= 100) {
            condition = assignments[4];
            recommendedCourse = recommendation[4];
        } else {
            System.out.print("System error");
        }
    }
    
    @Override
    public void start(Stage primaryStage) {
        BorderPane root = new BorderPane();
        root.setPadding(new Insets(10));

        VBox optionsBox = new VBox(10);
        optionsBox.setPadding(new Insets(10));
        optionsBox.setAlignment(Pos.CENTER_LEFT);

        Label instructionLabel = new Label("Select a Category:");
        optionsBox.getChildren().add(instructionLabel);

        ToggleGroup categoryGroup = new ToggleGroup();

        RadioButton railGeometryOption = new RadioButton("Rail Geometry");
        railGeometryOption.setToggleGroup(categoryGroup);
        railGeometryOption.setOnAction(e -> openMenuSelection(primaryStage));

        RadioButton sleeperOption = new RadioButton("Sleeper");
        sleeperOption.setToggleGroup(categoryGroup);
        sleeperOption.setOnAction(e -> primaryStage.close());

        RadioButton railOption = new RadioButton("Rail");
        railOption.setToggleGroup(categoryGroup);
        railOption.setOnAction(e -> primaryStage.close());

        optionsBox.getChildren().addAll(railGeometryOption, sleeperOption, railOption);

        root.setCenter(optionsBox);

        Scene scene = new Scene(root, 300, 200);
        primaryStage.setScene(scene);
        primaryStage.setTitle("Select a Category");
        primaryStage.show();
    }

    private void openMenuSelection(Stage primaryStage) {
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
            } else {
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

    private void configureInputField(TextField textField) {
        textField.setPromptText("Enter a positive number");
    }

    private void configurePositiveInputField(TextField textField) {
        textField.setPromptText("Enter a positive number");
    }

    private void openResultWindow(float exceedL, float exceedA, float exceedG) {
        resultStage = new Stage();
        BorderPane resultPane = new BorderPane();
        resultPane.setPadding(new Insets(10));

        GridPane gridPane = new GridPane();
        gridPane.setPadding(new Insets(10));
        gridPane.setHgap(10);
        gridPane.setVgap(10);
        gridPane.setAlignment(Pos.CENTER_LEFT);

        Label tgiResultLabel = new Label("TGI Result: " + String.format("%.2f", tgiOut) + "%");
        Label conditionLabel = new Label("Condition: " + condition);
        Label recommendationLabel = new Label("Recommended Course of Action: " + recommendedCourse);

        Label exceedLLabel = new Label("Longitudinal deviation over max: " + String.format("%.2f", exceedL));
        Label exceedALabel = new Label("Alignment deviation over max: " + String.format("%.2f", exceedA));
        Label exceedGLabel = new Label("Gauge deviation over max: " + String.format("%.2f", exceedG));

        gridPane.add(tgiResultLabel, 0, 0);
        gridPane.add(conditionLabel, 0, 1);
        gridPane.add(recommendationLabel, 0, 2);
        gridPane.add(exceedLLabel, 0, 3);
        gridPane.add(exceedALabel, 0, 4);
        gridPane.add(exceedGLabel, 0, 5);

        resultPane.setCenter(gridPane);
        Scene resultScene = new Scene(resultPane, 400, 300);
        resultStage.setScene(resultScene);
        resultStage.setTitle("TGI Result");
        resultStage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}