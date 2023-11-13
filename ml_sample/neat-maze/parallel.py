import multiprocessing as mp

import neat


class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self) -> bool:
        return False

    def _set_daemon(self, value) -> None:
        pass

    daemon = property(_get_daemon, _set_daemon)


class NonDaemonPool(mp.pool.Pool):
    def Process(self, *args, **kwds) -> mp.pool.Pool:
        proc = super(NonDaemonPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc

class EvaluatorParallel:
    def __init__(self, num_workers: int, decode_function, evaluate_function, revaluate: bool=False, timeout=None, parallel: bool=True, print_progress: bool=True):
        self.num_workers = num_workers
        self.decode_function = decode_function
        self.evaluate_function = evaluate_function
        self.revaluate = revaluate
        self.timeout = timeout
        self.parallel = parallel
        self.pool = NonDaemonPool(num_workers) if parallel and num_workers>0 else None
        self.print_progress = print_progress

    def __del__(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()

    def evaluate(self, genomes: dict, config: neat.Config, generation: int) -> None:

        size = len(genomes)

        if self.parallel:
            phenomes = {key: self.decode_function(genome, config.genome_config) for key,genome in genomes.items()}

            jobs: dict = {}
            for key, phenome in phenomes.items():
                # if already assinged fitness, skip evaluation
                if not self.revaluate and getattr(genomes[key], 'fitness', None) is not None:
                    continue

                args = (key, phenome, generation)
                jobs[key] = self.pool.apply_async(self.evaluate_function, args=args)

            # assign the result back to each genome
            for i,(key,genome) in enumerate(genomes.items()):
                if key not in jobs:
                    continue

                results: dict = jobs[key].get(timeout=self.timeout)
                for attr, data in results.items():
                    setattr(genome, attr, data)

                if self.print_progress:
                    print(f'\revaluating genomes ... {i+1: =4}/{size: =4}', end='')
            if self.print_progress:
                print('evaluating genomes ... done')

        else:
            for i,(key,genome) in enumerate(genomes.items()):
                phenome = self.decode_function(genome, config.genome_config)

                args = (key, phenome, generation)
                results = self.evaluate_function(*args)
                for attr, data in results.items():
                    setattr(genome, attr, data)
                if self.print_progress:
                    print(f'\revaluating genomes ... {i+1: =4}/{size: =4}', end='')
            if self.print_progress:
                print('evaluating genomes ... done')
