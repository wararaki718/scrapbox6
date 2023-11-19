import os
import warnings
from pathlib import Path
from configparser import ConfigParser
from typing import Optional

from neat import DefaultGenome


from neat.config import ConfigParameter, UnknownConfigItemError, Config


class MEConfig:
    """A simple container for user-configurable parameters of ME-NEAT."""

    __params = [
        ConfigParameter('offspring_size', int),
        ConfigParameter('fitness_criterion', str),
        ConfigParameter('fitness_threshold', float),
        ConfigParameter('no_fitness_termination', bool, False)
    ]

    def __init__(
        self,
        genome_type: DefaultGenome,
        filename: Path,
        custom_config: Optional[Config]=None
    ):
        # Check that the provided types have the required methods.
        assert hasattr(genome_type, 'parse_config')

        self.genome_type = genome_type

        if not filename.exists():
            raise Exception(f"No such config file: {filename.absolute()}")

        parameters = ConfigParser()
        with open(filename) as f:
            if hasattr(parameters, 'read_file'):
                parameters.read_file(f)
            else:
                parameters.readfp(f)

        if custom_config is not None:
            # [(section, key, value), ...]
            for cfg in custom_config:
                assert len(cfg) == 3, 'Invalid custom config input'
                section, key, value = cfg
                parameters[section][key] = str(value)

        # ME-NEAT configuration
        if not parameters.has_section('ME-NEAT'):
            raise RuntimeError("'ME-NEAT' section not found in ME-NEAT configuration file.")

        param_list_names = []
        for p in self.__params:
            if p.default is None:
                setattr(self, p.name, p.parse('ME-NEAT', parameters))
            else:
                try:
                    setattr(self, p.name, p.parse('ME-NEAT', parameters))
                except Exception:
                    setattr(self, p.name, p.default)
                    warnings.warn(
                        f"Using default {p.default} for '{p.name}'",
                        DeprecationWarning,
                    )
            param_list_names.append(p.name)

        param_dict = dict(parameters.items('ME-NEAT'))
        unknown_list = [x for x in param_dict if x not in param_list_names]

        if unknown_list:
            if len(unknown_list) > 1:
                raise UnknownConfigItemError("Unknown (section 'ME-NEAT') configuration items:\n" +
                                             "\n\t".join(unknown_list))
            raise UnknownConfigItemError(
                "Unknown (section 'ME-NEAT') configuration item {!s}".format(unknown_list[0]))

        # Parse type sections.
        genome_dict = dict(parameters.items(genome_type.__name__))
        self.genome_config = genome_type.parse_config(genome_dict)

    @classmethod
    def load(cls, config_file: Path, custom_config: Optional[Config]=None) -> "MEConfig":
        config = cls(
            DefaultGenome,
            config_file,
            custom_config=custom_config
        )
        return config
