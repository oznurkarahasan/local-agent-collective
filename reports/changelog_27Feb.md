feat 1: add orchestrator with parallel and sequential execution

- _plan() sends task to DeepSeek and receives JSON execution plan
- _execute_plan() resolves depends_on dependencies
- parallel execution for independent steps (Semaphore controlled)
- sequential execution for dependent steps
- _report() synthesizes all results into final report
- run() full end-to-end orchestration flow
- 18 unit tests, all passing"