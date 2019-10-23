import attr


@attr.s
class HubParams:
    batch_size = attr.ib(default=64)
    num_iterations = attr.ib(default=[20, 20])
    num_parts = attr.ib(default=2)
    context_size = attr.ib(default=7)
    num_evaluations = attr.ib(default=10)
    reverse = attr.ib(default=False)
    shuffle_docs = attr.ib(default=False)
    num_types = attr.ib(default=4096)
    corpus_name = attr.ib(default='childes-20180319')
    probes_name = attr.ib(default='childes-20180319_4096')