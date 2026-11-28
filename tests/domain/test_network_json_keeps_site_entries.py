"""A site network's JSON keeps its utility balance and level targets.

A network's ``source_metadata`` is left out of its JSON, so a site network
read back from JSON lost its site utility balance and its utility level
targets, and network evolution then pooled it as a plain network (review
finding 37, ``G-S/roundtrip.py``).
"""

from __future__ import annotations

from OpenPinch.contracts.input import HeatExchangerNetworkSchema
from OpenPinch.contracts.synthesis.site import (
    UtilityLevelBalance,
    UtilitySystemBalance,
)
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork


def _site_network() -> HeatExchangerNetwork:
    balance = UtilitySystemBalance(
        balances=(
            UtilityLevelBalance(
                level="HPS",
                period_idx=0,
                use=100.0,
                generation=0.0,
                imported=100.0,
                letdown_in=0.0,
                letdown_out=0.0,
                dump=0.0,
                cost=1000.0,
            ),
        ),
        period_costs=(1000.0,),
        operating_cost=1000.0,
        hot_utility=(100.0,),
        cold_utility=(0.0,),
    )
    network = HeatExchangerNetwork.from_structure(
        recovery=[], heaters=["C1"], coolers=["H1"], hot_utility="HU", cold_utility="CU"
    )
    return network.model_copy(
        update={
            "source_metadata": {
                "site_utility_balance": balance,
                "utility_level_targets": {"HPS": [100.0]},
            }
        }
    )


def test_a_site_networks_json_keeps_its_balance_and_level_targets():
    network = _site_network()

    back = HeatExchangerNetwork.model_validate_json(network.model_dump_json())

    # The balance comes back as its JSON form (the domain cannot rebuild a
    # contract type); readers check only that a site network has one.
    assert back.source_metadata["site_utility_balance"] == (
        network.source_metadata["site_utility_balance"].model_dump(mode="json")
    )
    assert back.source_metadata["utility_level_targets"] == {"HPS": [100.0]}


def test_a_site_seeds_json_keeps_its_process_duties():
    # A site seed records each process's duties by utility; network
    # evolution evolves it within each process, and pools it without them.
    processes = {"Heating/": {"HPS": [100.0]}, "Cooling/": {"CW": [80.0]}}
    network = HeatExchangerNetwork.from_structure(
        recovery=[], heaters=["C1"], coolers=["H1"], hot_utility="HU", cold_utility="CU"
    ).model_copy(update={"source_metadata": {"site_processes": processes}})

    back = HeatExchangerNetwork.model_validate_json(network.model_dump_json())

    assert back.source_metadata["site_processes"] == processes


def test_the_networks_serialisation_schema_keeps_its_fields_and_site():
    # A model-level serialiser collapsed the schema to an open object
    # (Codex review on #571); typed clients read the network's shape from it.
    properties = HeatExchangerNetwork.model_json_schema(mode="serialization")[
        "properties"
    ]

    assert {"exchangers", "total_annual_cost", "site"} <= set(properties)


def test_the_input_schema_takes_a_site_networks_json():
    # A problem input's network passes through HeatExchangerNetworkSchema,
    # which refused the site entries as an extra field (Codex review on
    # #571).
    network = _site_network()

    schema = HeatExchangerNetworkSchema.model_validate(network.model_dump(mode="json"))
    back = HeatExchangerNetwork.model_validate(schema.model_dump(mode="json"))

    assert back.source_metadata["utility_level_targets"] == {"HPS": [100.0]}
