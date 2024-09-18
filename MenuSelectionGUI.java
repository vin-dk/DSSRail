import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Label;
import javafx.scene.control.RadioButton;
import javafx.scene.control.ToggleGroup;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

public class MenuSelectionGUI extends Application {

    private String userChoice = "None";

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
        railGeometryOption.setOnAction(e -> {
            RailGeometry railGeometry = new RailGeometry();
            railGeometry.openMenuSelection(primaryStage);
        });

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

    public static void main(String[] args) {
        launch(args);
    }
}