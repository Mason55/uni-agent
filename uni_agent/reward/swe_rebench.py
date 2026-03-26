from pydantic import BaseModel


class SWEREBenchVerifySpec(BaseModel):
    metadata: dict

    @property
    def instance_id(self):
        return self.metadata["instance_id"]

    @property
    def gold_patch(self):
        return self.metadata["patch"]

    @property
    def eval_script(self):
        pass

    def _get_logs_eval(self, eval_output: str):
        pass

    def get_eval_report(self, eval_output: str):
        pass
