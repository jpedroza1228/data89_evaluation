library(tidyverse)
library(mirt)

y <- read_csv(here::here("data/quiz_data/quiz2_ready_irt.csv")) |> select(-1)
y2 <- read_csv(here::here("data/quiz_data/quiz2_retake_ready_irt.csv")) |> select(-1)

colnames(y)

# gt::gt(y |> rowid_to_column() |> select(rowid, disttable_submit2))

y <- y |> 
  mutate(
    denscomp_submit1 = if_else(denscomp_submit1 == "null", "-99", denscomp_submit1),
    denscomp_submit2 = if_else(denscomp_submit2 == "null", "-99", denscomp_submit1),
     across(
      c(
        disttable_submit2,
        denscomp_submit1,
        denscomp_submit2
      ),
      ~as.numeric(.x) |> round(2)
    )
  )

y |> count(denscomp_submit1)

y <- y |> 
  mutate(
    item1 = if_else(def_submit1 == def_true1, 1, 0),
    item2 = if_else(def_submit2 == def_true2, 1, 0),
    item3 = if_else(def_submit3 == def_true3, 1, 0),
    item4 = if_else(min2_submit1 == min2_true1, 1, 0),
    item5 = if_else(min2_submit2 == min2_true2, 1, 0),
    item6 = if_else(disttable_submit1 == disttable_true1, 1, 0),
    item7 = if_else(disttable_submit2 == disttable_true2, 1, 0),
    item8 = if_else(vispmf_submit1 == vispmf_true1, 1, 0),
    item9 = if_else(vispmf_submit2 == vispmf_true2, 1, 0),
    item10 = if_else(iddist_submit1 == iddist_true1, 1, 0),
    item11 = if_else(iddist_submit2 == iddist_true2, 1, 0),
    item12 = if_else(iddist_submit3 == iddist_true3, 1, 0),
    item13 = if_else(iddist_submit4 == iddist_true4, 1, 0),
    item14 = if_else(alice_submit1 == alice_true1, 1, 0),
    item15 = if_else(denscomp_submit1 == denscomp_true1, 1, 0),
    item16 = if_else(denscomp_submit2 == denscomp_true2, 1, 0)
  )

y_item <- y |> 
  select(
    matches("item")
    )


y2 <- y2 |> 
  mutate(
    across(
      c(
        disttable_submit2,
        denscomp_submit1,
        denscomp_submit2
      ),
      ~as.numeric(.x) |> round(2)
    )
  )

y2 <- y2 |> 
  mutate(
    item1 = if_else(def_submit1 == def_true1, 1, 0),
    item2 = if_else(def_submit2 == def_true2, 1, 0),
    item3 = if_else(def_submit3 == def_true3, 1, 0),
    item4 = if_else(sum2_submit1 == sum2_true1, 1, 0),
    item5 = if_else(sum2_submit2 == sum2_true2, 1, 0),
    item6 = if_else(disttable_submit1 == disttable_true1, 1, 0),
    item7 = if_else(disttable_submit2 == disttable_true2, 1, 0),
    item8 = if_else(vispmf_submit1 == vispmf_true1, 1, 0),
    item9 = if_else(vispmf_submit2 == vispmf_true2, 1, 0),
    item10 = if_else(dice_submit1 == dice_true1, 1, 0),
    item11 = if_else(dice_submit2 == dice_true2, 1, 0),
    item12 = if_else(dice_submit3 == dice_true3, 1, 0),
    item13 = if_else(dice_submit4 == dice_true4, 1, 0),
    item14 = if_else(marathon_submit1 == marathon_true1, 1, 0),
    item15 = if_else(marathon_submit2 == marathon_true2, 1, 0),
    item16 = if_else(denscomp_submit1 == denscomp_true1, 1, 0),
    item17 = if_else(denscomp_submit2 == denscomp_true2, 1, 0)
  )

y2_item <- y2 |> 
  select(
    matches("item")
    )





set.seed(122891)
irt_1pl <- mirt(y_item,
                1,
                itemtype = "1PL"
                )
irt_1pl

set.seed(122891)
irt_2pl <- mirt(y_item,
                1,
                itemtype = "2PL"
                )

set.seed(122891)
irt_3pl <- mirt(y_item,
                1,
                itemtype = "3PL"
                )

anova(irt_1pl, irt_2pl)
anova(irt_1pl, irt_3pl)
anova(irt_2pl, irt_3pl)

# Simplify = TRUE makes the output a clean table
param <- coef(irt_1pl, IRTpars = TRUE, simplify = TRUE)
param$items

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
  labs(title = "Difficulty (b) Parameter",
       subtitle = "For Quiz 2") +
  theme_light()


set.seed(122891)
theta_scores <- fscores(irt_1pl, method = "MAP")

set.seed(122891)
results <- data.frame(y_item, theta = theta_scores)
results <- results |> rename(theta = F1)
results$irt_t <- results$theta * 10 + 50
results <- results |> mutate(perc_irt = pnorm(theta))

results |> 
  ggplot(
    aes(
      theta
    )
  ) +
  geom_density(
    color = "black",
    fill = "seagreen",
    alpha = .5
  ) +
  labs(title = "Student Ability For Quiz 2") +
  theme_light()

results |> 
  ggplot(
    aes(
      perc_irt
    )
  ) +
  geom_density(
    color = "black",
    fill = "seagreen",
    alpha = .5
  ) +
  scale_x_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, .1)
  ) +
  labs(title = 'Percentile Ranking For Quiz 2') +
  theme_light()

# using retake data
set.seed(122891)
irt2_1pl <- mirt(y2_item,
                1,
                itemtype = "1PL"
                )

param2 <- coef(irt2_1pl, IRTpars = TRUE, simplify = TRUE)
param2$items

as_tibble(param2$items) |> 
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
  labs(title = "Difficulty (b) Parameter",
       subtitle = "For Quiz 2 Retake") +
  theme_light()

set.seed(122891)
theta2_scores <- fscores(irt2_1pl, method = "MAP")

set.seed(122891)
results2 <- data.frame(y2_item, theta = theta2_scores)
results2 <- results2 |> rename(theta = F1)
results2$irt_t <- results2$theta * 10 + 50
results2 <- results2 |> mutate(perc_irt = pnorm(theta))

results2 |> 
  ggplot(
    aes(
      theta
    )
  ) +
  geom_density(
    color = "black",
    fill = "seagreen",
    alpha = .5
  ) +
  labs(title = "Student Ability For Quiz 2 Retake") +
  theme_light()

results2 |> 
  ggplot(
    aes(
      perc_irt
    )
  ) +
  geom_density(
    color = "black",
    fill = "seagreen",
    alpha = .5
  ) +
  scale_x_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, .1)
  ) +
  labs(title = 'Percentile Ranking for Quiz 2 Retake') +
  theme_light()

results |> 
  summarize(
    avg = mean(perc_irt, na.rm = TRUE),
    med = median(perc_irt, na.rm = TRUE),
    sd = sd(perc_irt, na.rm = TRUE),
    across(
      c(
        avg,
        med,
        sd
      ),
      ~round(.x, 2)
    )
  )

results2 |> 
  summarize(
    avg = mean(perc_irt, na.rm = TRUE),
    med = median(perc_irt, na.rm = TRUE),
    sd = sd(perc_irt, na.rm = TRUE),
    across(
      c(
        avg,
        med,
        sd
      ),
      ~round(.x, 2)
    )
  )
