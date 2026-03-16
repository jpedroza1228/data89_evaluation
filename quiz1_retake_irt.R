library(tidyverse)
library(mirt)

y <- read_csv(here::here("data/quiz_data/q1_retake_clean.csv")) |> 
  select(-c(1:2))
y |> head()

y_sub <-
  y |> 
  select(
    -c(
      item2,
      item16,
      item20
    )
  )

set.seed(122891)
irt_1pl <- mirt(y_sub,
                1,
                itemtype = "1PL"
                )
irt_1pl
summary(irt_1pl)

# Simplify = TRUE makes the output a clean table
param <- coef(irt_1pl, IRTpars = TRUE, simplify = TRUE)
param$items
# gt::gt(as_tibble(param$items) |> round(3))

as_tibble(param$items) |> 
  select(b) |> 
  rowid_to_column() |> 
  pivot_longer(-rowid) |> 
  ggplot(
    aes(
      as.factor(rowid),
      value
    )
  ) +
  geom_col(
    position = position_dodge(),
    color = "black",
    fill = "seagreen"
  ) +
  geom_hline(
    yintercept = 0,
    color = "black",
    linewidth = 1) +
  facet_wrap(
    vars(
      name
    ),
    scales = "free"
  ) +
  theme_light()

set.seed(122891)
theta_scores <- fscores(irt_1pl, method = "MAP")

theta_scores[1:5,]

# 2. Convert to a data frame and add to your original data
set.seed(122891)
results <- data.frame(y_sub, theta = theta_scores)
# View the top of the results
results <- results |> rename(theta = F1)
results$irt_t <- results$theta * 10 + 50
results <- results |> mutate(perc_irt = pnorm(theta))

set.seed(122891)
results <- results |> 
  mutate(
    total = rowSums(across(matches("^item"))), .keep = "all",
    z_score = scale(total),
    t_score = z_score * 10 + 50,
    perc_rank = pnorm(z_score)
  )

jpcolor <- "seagreen"

results |> 
  pivot_longer(
    cols = c(
      theta,
      z_score
    )
  ) |> 
  ggplot(
    aes(
      value,
      name
    )
  ) +
  ggridges::geom_density_ridges(
    aes(
      fill = name
    ),
    alpha = .5
  ) +
  geom_vline(
    xintercept = 3,
    linetype = "dashed",
    color = jpcolor
  ) +
  geom_vline(
    xintercept = -3,
    linetype = "dashed",
    color = jpcolor
  ) +
  theme_light() +
  theme(
    legend.position = "none"
  )

results |> 
  pivot_longer(
    cols = c(
      irt_t,
      t_score
    )
  ) |> 
  ggplot(
    aes(
      value,
      name
    )
  ) +
  ggridges::geom_density_ridges(
    aes(
      fill = name
    ),
    alpha = .5
  ) +
  geom_vline(
    xintercept = 20,
    linetype = "dashed",
    color = jpcolor
  ) +
  geom_vline(
    xintercept = 80,
    linetype = "dashed",
    color = jpcolor
  ) +
  theme_light() +
  theme(
    legend.position = "none"
  )

results |> 
  pivot_longer(
    cols = c(
      perc_irt,
      perc_rank
    )
  ) |> 
  ggplot(
    aes(
      value,
      name
    )
  ) +
  ggridges::geom_density_ridges(
    aes(
      fill = name
    ),
    alpha = .5
  ) +
  scale_x_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, .1)
  ) +
  theme_light() +
  theme(
    legend.position = "none"
  )



set.seed(122891)
irt_2pl <- mirt(y_sub,
                1,
                itemtype = "2PL"
                )
irt_2pl
summary(irt_2pl)

# Simplify = TRUE makes the output a clean table
param2 <- coef(irt_2pl, IRTpars = TRUE, simplify = TRUE)
param2$items
# gt::gt(as_tibble(param$items) |> round(3))

as_tibble(param2$items) |> 
  select(a, b) |> 
  rowid_to_column() |> 
  pivot_longer(-rowid) |> 
  ggplot(
    aes(
      as.factor(rowid),
      value
    )
  ) +
  geom_col(
    position = position_dodge(),
    color = "black",
    fill = "seagreen"
  ) +
  geom_hline(
    yintercept = 0,
    color = "black",
    linewidth = 1) +
  facet_wrap(
    vars(
      name
    ),
    scales = "free"
  ) +
  theme_light()

set.seed(122891)
theta_scores2 <- fscores(irt_2pl, method = "MAP")

# 2. Convert to a data frame and add to your original data
set.seed(122891)
results2 <- data.frame(y_sub, theta = theta_scores2)
# View the top of the results
results2 <- results2 |> rename(theta = F1)
results2$irt_t <- results2$theta * 10 + 50
results2 <- results2 |> mutate(perc_irt = pnorm(theta))

set.seed(122891)
results2 <- results2 |> 
  mutate(
    total = rowSums(across(matches("^item"))), .keep = "all",
    z_score = scale(total),
    t_score = z_score * 10 + 50,
    perc_rank = pnorm(z_score)
  )

results2 |> 
  pivot_longer(
    cols = c(
      theta,
      z_score
    )
  ) |> 
  ggplot(
    aes(
      value,
      name
    )
  ) +
  ggridges::geom_density_ridges(
    aes(
      fill = name
    ),
    alpha = .5
  ) +
  geom_vline(
    xintercept = 3,
    linetype = "dashed",
    color = jpcolor
  ) +
  geom_vline(
    xintercept = -3,
    linetype = "dashed",
    color = jpcolor
  ) +
  theme_light() +
  theme(
    legend.position = "none"
  )

results2 |> 
  pivot_longer(
    cols = c(
      irt_t,
      t_score
    )
  ) |> 
  ggplot(
    aes(
      value,
      name
    )
  ) +
  ggridges::geom_density_ridges(
    aes(
      fill = name
    ),
    alpha = .5
  ) +
  geom_vline(
    xintercept = 20,
    linetype = "dashed",
    color = jpcolor
  ) +
  geom_vline(
    xintercept = 80,
    linetype = "dashed",
    color = jpcolor
  ) +
  theme_light() +
  theme(
    legend.position = "none"
  )

results2 |> 
  pivot_longer(
    cols = c(
      perc_irt,
      perc_rank
    )
  ) |> 
  ggplot(
    aes(
      value,
      name
    )
  ) +
  ggridges::geom_density_ridges(
    aes(
      fill = name
    ),
    alpha = .5
  ) +
  scale_x_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, .1)
  ) +
  theme_light() +
  theme(
    legend.position = "none"
  )

results2 |> 
  summarize(
    across(
      c(
        perc_rank,
        perc_irt
      ),
      list(
        avg = ~mean(.x, na.rm = TRUE),
        med = ~median(.x, na.rm = TRUE),
        sd = ~sd(.x, na.rm = TRUE)
      )
  )
  ) |> 
  t()
