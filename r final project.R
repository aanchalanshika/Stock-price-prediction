install.packages("shiny")
install.packages("quantmod")
install.packages("TTR")
install.packages("forecast")
install.packages("caret")
install.packages("e1071")
install.packages("rpart")
install.packages("class")

library(shiny)
library(quantmod)
library(TTR)
library(forecast)
library(caret)
library(e1071)
library(rpart)
library(class)
library(shiny)
library(ggplot2)

# Define UI
ui <- fluidPage(
  titlePanel("Stock Price Prediction Dashboard"),
  sidebarLayout(
    sidebarPanel(
      h3("Model Parameters"),
      sliderInput("k_value", "K-Nearest Neighbors (K):", min = 1, max = 20, value = 5),
      checkboxGroupInput(
        "models",
        "Select Models to Display:",
        choices = c("SVM" = "svm", "Decision Tree" = "tree", "KNN" = "knn"),
        selected = c("svm", "tree", "knn")
      )
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Plot", plotOutput("predictionPlot")),
        tabPanel("Metrics", tableOutput("metricsTable"))
      )
    )
  )
)

# Define Server
server <- function(input, output) {
  # Data Preparation and Model Training
  getSymbols("AAPL", src = "yahoo", from = "2021-01-01", to = "2023-01-01")
  aapl_data <- na.omit(AAPL)
  aapl_data$SMA_20 <- SMA(Cl(aapl_data), n = 20)
  aapl_data$SMA_50 <- SMA(Cl(aapl_data), n = 50)
  aapl_data$RSI_14 <- RSI(Cl(aapl_data), n = 14)
  aapl_data <- na.omit(aapl_data)
  
  set.seed(123)
  train_index <- createDataPartition(y = Cl(aapl_data), p = 0.8, list = FALSE, times = 1)
  train_data <- aapl_data[train_index, ]
  test_data <- aapl_data[-train_index, ]
  
  actual_values <- as.numeric(Cl(test_data))
  
  svm_train <- data.frame(
    Close = as.numeric(Cl(train_data)),
    SMA_20 = as.numeric(train_data$SMA_20),
    SMA_50 = as.numeric(train_data$SMA_50),
    RSI_14 = as.numeric(train_data$RSI_14)
  )
  svm_test <- data.frame(
    Close = as.numeric(Cl(test_data)),
    SMA_20 = as.numeric(test_data$SMA_20),
    SMA_50 = as.numeric(test_data$SMA_50),
    RSI_14 = as.numeric(test_data$RSI_14)
  )
  
  svm_model <- svm(Close ~ ., data = svm_train, kernel = "radial")
  svm_pred <- predict(svm_model, svm_test)
  
  tree_model <- rpart(Close ~ ., data = svm_train)
  tree_pred <- predict(tree_model, svm_test)
  
  # Reactive Model Predictions
  reactive_predictions <- reactive({
    k <- input$k_value
    knn_pred <- knn(
      train = svm_train[, -1],
      test = svm_test[, -1],
      cl = svm_train$Close,
      k = k
    )
    list(
      svm = as.numeric(svm_pred),
      tree = as.numeric(tree_pred),
      knn = as.numeric(knn_pred)
    )
  })
  
  # Plot Predictions
  output$predictionPlot <- renderPlot({
    predictions <- reactive_predictions()
    selected_models <- input$models
    plot(actual_values, type = "l", col = "black", xlab = "Time", ylab = "Close Price", main = "Model Predictions vs Actual")
    if ("svm" %in% selected_models) lines(predictions$svm, col = "green")
    if ("tree" %in% selected_models) lines(predictions$tree, col = "red")
    if ("knn" %in% selected_models) lines(predictions$knn, col = "purple")
    legend(
      "topright",
      legend = c("Actual", selected_models),
      col = c("black", "green", "red", "purple")[1:length(selected_models) + 1],
      lty = 1
    )
  })
  
  # Error Metrics Table
  output$metricsTable <- renderTable({
    predictions <- reactive_predictions()
    data.frame(
      Model = c("SVM", "Decision Tree", "KNN"),
      MAE = c(
        mean(abs(predictions$svm - actual_values)),
        mean(abs(predictions$tree - actual_values)),
        mean(abs(predictions$knn - actual_values))
      ),
      RMSE = c(
        sqrt(mean((predictions$svm - actual_values)^2)),
        sqrt(mean((predictions$tree - actual_values)^2)),
        sqrt(mean((predictions$knn - actual_values)^2))
      )
    )
  })
}

# Run App
shinyApp(ui = ui, server = server)

