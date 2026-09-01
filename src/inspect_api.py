from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
print([n for n in dir(Recording) if 'embed' in n.lower() or 'analyze' in n.lower()])
print([n for n in dir(Analyzer) if 'embed' in n.lower() or 'analyze' in n.lower()])
