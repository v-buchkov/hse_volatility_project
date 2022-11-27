from enum import Enum


class SamplingSchemas(Enum):
    """
    Here we save all important sampling schemas that we tried to simply log all models and info
    that we sample for higher level features extraction
    """
    TEST_SCHEMA = 'testSchema'
    MDS_SCHEMA = 'mdsSchema'
    SLOW_MDS_SCHEMA = 'slowMdsSchema'
    IMPACT_SCHEMA = 'ForImpactEst|200ms|4levels'
    FIRST_PRICE_PREDICTION_SCHEMA = 'PricePred|1000ms|10levels'
    FILTERED_FIRST_PRICE_PREDICTION_SCHEMA = 'FilteredPricePred|1000ms|10levels'
    OPTIONS_SCHEMA_USDRUB = "OptionsInit|3kopeks"
    G10_PRICE_PREDICTION_SCHEMA = "PricePred|50ms|0levels"
    REPORT_SCHEMA = "PricePred|100ms|4levels"
    SLOW_SWAPS_SCHEMA = "SwapsMoexBook|10000ms"
    SLOW_BONDS_SCHEMA = "PricePred|60000ms"
    HOURLY_BONDS_SCHEMA = "PricePred|60000ms|"
    LOW_LATENCY_SCHEMA = "LowLatency|1ms|0levels"
    LOW_LATENCY_SCHEMA_500US = "LowLatency|500us|0levels"
    CME_FUTURES_SCHEMA = "PricePred|500us"
    HIGH_RESOLUTION_SCHEMA_50US = "HighResolutionLatency|50us|0levels"
    HIGH_RESOLUTION_SCHEMA_100US = "HighResolutionLatency|100us|0levels"
    HIGH_RESOLUTION_SCHEMA_200US = "HighResolutionLatency|200us|0levels"
    EVERY_KOPEK_SCHEMA = 'Options|1kopeks'
    EVERY_TICK_SCHEMA = 'EveryTick|0ms'
    EVERY_MINUTE_SCHEMA = 'PricePred|60s'
    HOURLY_SCHEMA = "PricePred|1h"
    EVERY_2H_SCHEMA = "PricePred|2h"
    EVERY_3H_SCHEMA = "PricePred|3h"
    EVERY_6H_SCHEMA = "PricePred|6h"
    EVERY_8H_SCHEMA = "PricePred|8h"
