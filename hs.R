library(tidyverse)
library(ggplot2)
library(rstanarm)
library(shinystan)
library(bayesplot)
library(brms)

df = bind_cols(read_csv("X.csv") %>% select(starts_with("corr")) %>% select(-1),
               read_csv("y.csv") %>% rename(y = clex_mut) %>% select(y))

# I set the treedepth max here low to make the model run, but it isn't
# recommended
fit = stan_glm(paste(names(df)[-190], collapse = " + ") %>% paste0("y ~ ", .),
               data = df,
               prior = hs(global_scale = 20.0 / (170.0 * sqrt(nrow(df)))),
               chains = 4, cores = 4,
               control = list(max_treedepth = 8))

mcmc_intervals(fit %>% as.matrix)

posterior_interval(fit) %>% as.tibble %>%
  mutate(rn = row_number()) %>%
  rename(low = '5%', high = '95%') %>%
  ggplot(aes(rn)) +
  geom_errorbar(aes(ymin = low, ymax = high)) +
  ylab("Parameter values") +
  xlab("Parameter number (0 is intercept)")

# fit2 = brm(paste(names(df)[-190], collapse = " + ") %>% paste0("y ~ ", .),
#            data = df,
#            prior = prior(horseshoe(scale_global = 0.004683455)),
#            chains = 1, cores = 4,
#            control = list(max_treedepth = 10))
# 
# make_stancode(paste(names(df)[-190], collapse = " + ") %>% paste0("y ~ ", .),
#               data = df,
#               prior = prior(horseshoe(scale_global = 0.004683455)))
# 
# mcmc_intervals(as.matrix(fit2)[,1:190])
