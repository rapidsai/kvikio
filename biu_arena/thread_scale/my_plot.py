import matplotlib.pyplot as plt


class PlotManager:
    def __init__(self):
        self.num_threads = [1, 2, 4, 8, 16, 32, 64]
        self.BS_threadpool_compute_strong_scaling = [322,
                                                     165,
                                                     83.0,
                                                     43.1,
                                                     30.8,
                                                     54.0,
                                                     63.4]
        self.BS_threadpool_compute_weak_scaling = [32.8,
                                                   33.1,
                                                   33.4,
                                                   34.6,
                                                   44.0,
                                                   173,
                                                   397]
        self.simple_threadpool_compute_strong_scaling = [322,
                                                         167,
                                                         85.8,
                                                         46.3,
                                                         27.1,
                                                         20.7,
                                                         31.3]
        self.simple_threadpool_compute_weak_scaling = [32.3,
                                                       32.6,
                                                       37.2,
                                                       37.8,
                                                       39.4,
                                                       47.4,
                                                       187]
        self.static_task_compute_strong_scaling = [325,
                                                   166,
                                                   88.0,
                                                   48.5,
                                                   30.1,
                                                   21.0,
                                                   18.8]
        self.static_task_compute_weak_scaling = [40.3,
                                                 39.9,
                                                 40.5,
                                                 40.5,
                                                 41.1,
                                                 44.4,
                                                 53.8]

    def do_it(self):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        f.suptitle("AMD EPYC 7642 48-Core Processor")

        for ax in [ax1, ax2]:
            if ax == ax1:
                bs = self.BS_threadpool_compute_strong_scaling
                simple = self.simple_threadpool_compute_strong_scaling
                static = self.static_task_compute_strong_scaling
                title = "strong scaling"
            else:
                bs = self.BS_threadpool_compute_weak_scaling
                simple = self.simple_threadpool_compute_weak_scaling
                static = self.static_task_compute_weak_scaling
                title = "weak scaling"

            ax.plot(self.num_threads, bs, marker='o',
                    label="BS threadpool")
            ax.plot(self.num_threads, simple, marker='o',
                    label="simple threadpool")
            ax.plot(self.num_threads, static, marker='o',
                    label="static tasks")
            ax.set_title(title)
            ax.set_yscale("log", base=2)
            ax.set_xscale("log", base=2)
            ax.set_xlabel('thread count')
            ax.set_ylabel('time [ms]')
            ax.set_xticks(self.num_threads)
            ax.set_xticklabels(self.num_threads)
            ax.grid(linestyle='--', color='#000000')
            ax.legend(edgecolor="#000000")

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    pm = PlotManager()
    pm.do_it()
