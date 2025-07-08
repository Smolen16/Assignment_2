args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript linear_model.r <filename> <x_column> <y_column>")
}

filename <- args[1]
x_col <- args[2]
y_col <- args[3]

data <- read.csv("regression_data.csv")
formula <- as.formula(paste(y_col, "~", x_col))
model <- lm(formula, data = data)

library(ggplot2)
plot <- ggplot(data, aes_string(x = x_col, y = y_col)) +
  geom_point(color = "red") +
  geom_smooth(method = "lm", color = "blue") +
  ggtitle(paste(y_col, "vs", x_col)) +
  xlab(x_col) +
  ylab(y_col)

ggsave("linear_regression_r_output.png", plot)
print(plot)

dataset <- read.csv("regression_data.csv") # Reading in the csv file

print(dataset) # Double Checking the dataset to make sure everything worked.

plot(dataset$YearsExperience, dataset$Salary, col="red")

model <- lm(Salary ~ YearsExperience, data=dataset) # Making a linear model of the dataset in order to run a regression analysis

library(ggplot2) # Installing ggplot2 in order to run the regression.

library(ggplot2)
ggplot() +
    geom_point(aes(x = dataset$YearsExperience, y = dataset$Salary), colour = 'blue') +
    geom_line(aes(x = dataset$YearsExperience, y = predict(model, newdata = dataset)), colour = 'black') +
    ggtitle('Salary vs Experience') +
    xlab('Years of Experience') +
    ylab('Salary')

summary(model)

library(ggplot2)

# Data
df <- data.frame(
  x = c(dataset$YearsExperience),
  y = c(dataset$Salary)
)

# Fit model
model <- lm(y ~ x, data = df)
slope <- coef(model)[2]
intercept <- coef(model)[1]
r <- cor(df$x, df$y)
pred <- predict(model)
mse <- mean((df$y - pred)^2)

# Plot
ggplot(df, aes(x = x, y = y)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  annotate("text", x = 1.5, y = max(df$y) - 0.5,
           label = paste("y =", round(slope, 2), "x +", round(intercept, 2),
                         "\nr =", round(r, 2), "\nMSE =", round(mse, 2)),
           size = 4) +
  labs(title = "Linear Fit",
       x = "x", y = "y") +
  theme_minimal()

ggsave("regression_plot_r.png")


