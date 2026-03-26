from pydantic import BaseModel, ConfigDict


class TemplateConfig(BaseModel):
    system_template: str = ""
    instance_template: str = ""
    model_config = ConfigDict(extra="forbid")

    def get_system_prompt(self) -> str:
        return self.system_template
