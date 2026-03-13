from pydantic import BaseModel


class ForecastFeatures(BaseModel):

    Store: int
    Promo: int
    SchoolHoliday: int
    StateHoliday: str
    StoreType: str
    Assortment: str
    CompetitionDistance: float
    Promo2: int

    month: int
    day: int
    week_of_year: int
    day_of_week: int
    is_weekend: int
    is_month_start: int
    is_month_end: int

    lag_1: float
    lag_7: float
    lag_14: float
    lag_28: float

    rolling_mean_7: float
    rolling_mean_14: float
    rolling_std_7: float