#      The Certora Prover
#      Copyright (C) 2025  Certora Ltd.
#
#      This program is free software: you can redistribute it and/or modify
#      it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, version 3 of the License.
#
#      This program is distributed in the hope that it will be useful,
#      but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#      GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License
#      along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import List
from langchain_core.messages import BaseMessage, AnyMessage
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.runnables import Runnable

def cached_invoke(b: Runnable[LanguageModelInput, BaseMessage], s: List[AnyMessage]) -> BaseMessage:
    """
    Send messages `s` to the llm `b` after adding caching instructions.
    """
    ...

async def acached_invoke(b: Runnable[LanguageModelInput, BaseMessage], s: List[AnyMessage]) -> BaseMessage:
    """
    Send messages `s` to the llm `b` after adding caching instructions.
    """
    ...