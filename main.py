import yaml
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ETF_NAME = "name"
ETF_PROJECTED_GROWTH = "projected_growth"
ETF_HOLDING_FEE = "holding_fee"

STRATEGY_NAME = "name"
STRATEGY_YEARLY_INVESTMENTS = "yearly_investments"
STRATEGY_ETF_COMPONENTS = "etf_components"


class InvestmentStrategy:

    @staticmethod
    def _init_etf_components(etf_components):
        return {c: 0 for c in etf_components}

    def __init__(self, name, yearly_investments, etf_components):
        self.name = name
        self.yearly_investments = yearly_investments
        if type(etf_components) is list:
            self.etf_components = self._init_etf_components(etf_components)
        elif type(etf_components) is dict:
            self.etf_components = etf_components
        else:
            raise TypeError("etf components for investment strategy {} must be a list or dictionary".format(self.name))

        self.projected_growth = 0
        self._calc_projected_growth()

    def _calc_projected_growth(self):
        self.projected_growth = 1 + sum([(etf.projected_growth - etf.holding_fee) * weight for
                                         etf, weight in self.etf_components.items()]) / 100
        x = 9

    def project_assets(self, years, inflation_rate=0, only_gains=True):
        if not self.yearly_investments:
            raise(Exception("Cannot project assets for strategy {} since yearly investments list is empty".format(self.name)))

        projected_assets = [self.yearly_investments[0]]
        total_investment = projected_assets[0]
        projected_gains = [0]

        zero_padded_yearly_investments = list(self.yearly_investments) + max(0, years - len(self.yearly_investments)) * [0]

        for i in range(1, years):
            yearly_investment = zero_padded_yearly_investments[i]
            total_investment = total_investment + yearly_investment
            yearly_gain = projected_assets[i-1] * (self.projected_growth - inflation_rate) + yearly_investment

            projected_assets.append(yearly_gain)
            projected_gains.append(projected_assets[i] - total_investment)

        return projected_gains if only_gains else projected_assets


class ETF:
    def __init__(self, name, projected_growth, holding_fee):
        self.name = name
        self.projected_growth = projected_growth
        self.holding_fee = holding_fee

    def __hash__(self):
        random.seed(self.name)
        return random.randint(0, 1000000)

    def __eq__(self, other):
        return self.name == other.name

def line_style_iter():
    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
    shapes = ["o", "^", "*", "D"]
    for shape in shapes:
        for color in colors:
            yield "{}{}".format(color, shape)

def draw_chart(years, *strategies, inflation_rate=0, only_gains=True):
    time = np.arange(0, years, 1)
    projections = [strategy.project_assets(years, inflation_rate=inflation_rate, only_gains=only_gains) for
                        strategy in strategies]
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