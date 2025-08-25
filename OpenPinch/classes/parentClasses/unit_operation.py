
from abc import ABC, abstractmethod

class UnitOperation(ABC):
    """Represents an operation that results in a change to a stream (heatexchange/split/mix etc).
    energy_inputs: List of energy flows coming to this unit op
    energy_outputs: List of energy flows going from this unit op
    material_inputs: List of material flows coming to this unit op
    material_outputs: List of material flows going from this unit op
    """
    @abstractmethod
    def __init__(self):
        self.energy_inputs = []
        self.energy_outputs = []
        self.material_inputs = []
        self.material_outputs = []

    def add_energy_input(self, energy_input):
        self.energy_inputs.append(energy_input)

    def add_energy_output(self, energy_output):
        self.energy_outputs.append(energy_output)

    def add_material_input(self, material_input):
        self.material_inputs.append(material_input)

    def add_material_output(self, material_output):
        self.material_outputs.append(material_output)