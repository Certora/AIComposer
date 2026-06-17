from typing import Literal, TypeGuard
from dataclasses import dataclass, field

type ClaudeModelNames = Literal["opus", "sonnet", "haiku", "fable"]

@dataclass
class ModelFeatures:
    interleaved_thinking: bool
    adaptive_thinking: bool
    version_tuple: tuple[int, int]
    name: ClaudeModelNames

class NoSuchElementError(RuntimeError):
    pass

valid_names : set[ClaudeModelNames] = {"opus", "sonnet", "haiku", "fable"}

def _validate_model(s: str) -> TypeGuard[ClaudeModelNames]:
    return s in valid_names

_intearleaved_pivot_version = (4, 5)

@dataclass
class _ListIter[T]:
    l: list[T]
    ind: int = field(default=0)

    def has_next(self) -> bool:
        return self.ind < len(self.l)
    
    def peek(self) -> T:
        if not self.has_next():
            raise NoSuchElementError("Invalid state, no more elements")
        return self.l[self.ind]
    
    def next(self) -> T:
        if not self.has_next():
            raise NoSuchElementError("Invalid state, no more elements")
        to_ret = self.l[self.ind]
        self.ind += 1
        return to_ret

def model_parser(
    model_name: str
) -> ModelFeatures:
    stream = _ListIter(model_name.split("-"))
    parsing : Literal["claude", "model", "version"] = "claude"
    try:
        claude = stream.next()
        if claude != "claude":
            raise ValueError(f"Unrecognized model provider: {claude}")
        parsing = "model"
        model = stream.next()
        if _validate_model(model):
            model_class = model
        else:
            raise ValueError(f"Unrecognized model name: {model}")
        parsing = "version"
        
        major_version = int(stream.next())

        if stream.has_next():
            minor_version = int(stream.next())
        else:
            minor_version = 0
        
        version_tuple = (major_version, minor_version)
        interleaved_flag = version_tuple <= _intearleaved_pivot_version
        adaptive_flag = not interleaved_flag
        return ModelFeatures(
            interleaved_thinking=interleaved_flag,
            adaptive_thinking=adaptive_flag,
            name=model_class,
            version_tuple=version_tuple
        )
    except NoSuchElementError as exc:
        raise ValueError(f"Error parsing {parsing} from model identifier {model_name}; ran out of tokens") from exc
    except ValueError as exc:
        if parsing != "version":
            raise exc
        raise ValueError(f"Error parsing version from model identifier {model_name}; ill-formed version number") from exc