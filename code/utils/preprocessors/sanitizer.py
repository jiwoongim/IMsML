from abc_processor import Base 
import re


# this will filter out data to the regexp input into config.
class Processor(Base):
    def run(self, data):
        temp = re.sub(self.config, "", data)
        return temp
