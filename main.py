import yaml
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

ETF_NAME = "name"
ETF_PROJECTED_GROWTH = "projected_growth"
ETF_HOLDING_FEE = "holding_fee"

STRATEGY_NAME = "name"
STRATEGY_YEARLY_INVESTMENTS = "yearly_investments"
STRATEGY_ETF_COMPONENTS = "etf_components"


class EtfPortfolioValidationError(Exception):
    pass


@dataclass(frozen=True, eq=True)
class ETF:
    name: str
    projected_growth: float
    holding_fee: float


class EtfPortfolio:
    @staticmethod
    def _etf_portfolio_from_list(etf_components):
        return {c: 0 for c in etf_components}

    @staticmethod
    def _validate_input_types(etf_components):
        if type(etf_components) not in (list, dict):
            raise EtfPortfolioValidationError(f'An EtfPortfolio object cannot be instantiated from data of type {type(etf_components)}')

    @staticmethod
    def validate_portfolio_validity(portfolio):
        for etf, weight in portfolio:
            if type(etf) is not ETF:
                raise EtfPortfolioValidationError(f'each element in an etf portfolio must belong to the {ETF.__class__} class')
            if type(weight) not in (int, float):
                raise EtfPortfolioValidationError(f'the weight of each element in an etf portfolio must be an int or float')

        if sum(weight_in_portfolio for _, weight_in_portfolio in portfolio) != 1:
            raise ValueError("weights in an etf portfolio must sum up to 1!")

    def __init__(self, etf_components):
        self._validate_input_types(etf_components)

        self._portfolio_dict = etf_components if type(etf_components) is dict else self._etf_portfolio_from_list(etf_components)
        self.validate_portfolio_validity(self)

    def __iter__(self):
        return iter(self._portfolio_dict.items())


class InvestmentStrategy:
    def __init__(self, name, yearly_investments, etf_components):
        self.name = name
        self.yearly_investments = yearly_investments

        try:
            self.etf_portfolio = etf_components if type(etf_components) is EtfPortfolio else EtfPortfolio(etf_components)
        except EtfPortfolioValidationError as ex:
            raise EtfPortfolioValidationError(f'failed to create investment strategy {self.name} due to the following '\
                                              f'validation error while attempting to initialize the associated '
                                              f'portfolio: {str(ex)}')

        self.projected_growth = 0
        self._calc_projected_growth()

    def _calc_projected_growth(self):
        self.projected_growth = 1 + sum([(etf.projected_growth - etf.holding_fee) * weight for
                                         etf, weight in self.etf_portfolio]) / 100

    def project(self, years, inflation_rate=0, only_gains=True):
        projected_assets_by_year = [self.yearly_investments[0]] + [0] * (years - 1)
        projected_gains_by_year = [0] * years
        zero_padded_yearly_investments = list(self.yearly_investments) + max(0, years - len(self.yearly_investments)) * [0]

        total_investment = projected_assets_by_year[0]
        for i in range(1, years):
            invested_this_year = zero_padded_yearly_investments[i]
            total_investment = total_investment + invested_this_year
            total_assets = projected_assets_by_year[i-1] * (self.projected_growth - inflation_rate) + invested_this_year

            projected_assets_by_year[i] = total_assets
            projected_gains_by_year[i] = projected_assets_by_year[i] - total_investment

        return projected_gains_by_year if only_gains else projected_assets_by_year


def line_style_iter():
    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
    shapes = ["o", "^", "*", "D"]
    for shape in shapes:
        for color in colors:
            yield "{}{}".format(color, shape)


def draw_chart(years, *strategies, inflation_rate=0, only_gains=True):
    time = np.arange(0, years, 1)
    projections = \
        [strategy.project(years, inflation_rate=inflation_rate, only_gains=only_gains) for strategy in strategies]
    gain_text_component = "total yearly gains" if only_gains else "total assets worth"
    inflation_text_component = "inflation adjusted " if inflation_rate else ""
    y_label = inflation_text_component + gain_text_component
    title = inflation_text_component + gain_text_component + " per strategy"

    # init pyplot chart
    line_styles = line_style_iter()
    fig, ax = plt.subplots()

    # add each strategy projection to the chart
    for projection, strategy in zip(projections, strategies):
        ax.plot(time, projection, next(line_styles), label=strategy.name)

    ax.set(xlabel='time (years)', ylabel=y_label, title=title)
    plt.xticks(np.arange(0, years, 1))

    legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')
    legend.get_frame().set_facecolor('C0')

    plt.show()


def from_yaml(path):
    yaml_input = None
    with open(path, 'r') as stream:
        try:
            yaml_input = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return yaml_input


def init_etfs(etf_config):
    return {conf[ETF_NAME]: ETF(conf[ETF_NAME],
                                conf[ETF_PROJECTED_GROWTH],
                                conf[ETF_HOLDING_FEE])
            for conf in etf_config}


def init_strategies(strategy_config, etfs):
    strategies = [s for s in strategy_config]
    for s in strategies:
        s[STRATEGY_ETF_COMPONENTS] = {etfs[name]: weight for name, weight in s[STRATEGY_ETF_COMPONENTS].items()}

        if s[STRATEGY_YEARLY_INVESTMENTS] is str:
            investments = s[STRATEGY_YEARLY_INVESTMENTS].split("*")
            assert len(investments) == 2, "incorrect yearly investments format in strategy {}".format(s[STRATEGY_NAME])
            per_year = 0
            years = 1
            s[STRATEGY_YEARLY_INVESTMENTS] = [investments[per_year] for i in range(years)]

    return [InvestmentStrategy(s[STRATEGY_NAME],
                               s[STRATEGY_YEARLY_INVESTMENTS],
                               s[STRATEGY_ETF_COMPONENTS])
            for s in strategies]


def main():
    conf = from_yaml("data.yaml")
    etfs = init_etfs(conf["etfs"])
    strategies = init_strategies(conf["strategies"], etfs)
    draw_chart(43, *strategies, inflation_rate=0.02)


if __name__ == "__main__":
    main()