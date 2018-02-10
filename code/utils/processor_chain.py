class ProcessorChain():
    def __init__(self):
        self.processors = []

    def load(self, processor):
        self.processors.append(processor)

    def run(self, data):
        temp = data
        
        for processor in self.processors:
            temp = processor.run(data)

        return temp











