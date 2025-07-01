class Task:
    def __init__(self, name, action):
        self.name = name
        self.action = action

    def execute(self):
        print(f"Executing task: {self.name}")
        return self.action()

class Workflow:
    def __init__(self, tasks):
        assert len(tasks) == 3, "Workflow must contain exactly three tasks."
        self.task1, self.task2, self.task3 = tasks
        self.sequence_log = []

    def execute(self):
        print("\n[WORKFLOW INITIATED]\n")

        result1 = self.task1.execute()
        self.sequence_log.append(result1)

        result2 = self.task2.execute()
        self.sequence_log.append(result2)

        result3 = self.task3.execute()
        self.sequence_log.append(result3)

        # Final task
        final_result = f"Workflow completed with results: {self.sequence_log}"
        print(f"\n[COMPLETED] {final_result}")
        return final_result

# Define tasks
def task1_action():
    return "Task 1 Completed"

def task2_action():
    return "Task 2 Completed"

def task3_action():
    return "Task 3 Completed"

# Create tasks
task1 = Task("Task 1", task1_action)
task2 = Task("Task 2", task2_action)
task3 = Task("Task 3", task3_action)

# Create workflow and execute it
workflow = Workflow([task1, task2, task3])
workflow.execute()
