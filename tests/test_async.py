"""Tests for async/parallel processing functions."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest


class TestAsyncLLM:
    """Tests for async LLM wrapper."""

    @pytest.mark.asyncio
    async def test_call_llm_async_basic(self):
        """Test that call_llm_async calls acompletion correctly."""
        from ontoagain.llm import call_llm_async

        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "Test response"

        with patch("ontoagain.llm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            result = await call_llm_async(
                model="test-model",
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert result == "Test response"
            mock_acompletion.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_llm_async_with_api_base(self):
        """Test that API base is applied correctly."""
        from ontoagain.llm import call_llm_async

        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "Response"

        with patch("ontoagain.llm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            with patch("ontoagain.llm._get_api_base", return_value="https://custom.api"):
                result = await call_llm_async(
                    model="test-model",
                    messages=[{"role": "user", "content": "Hello"}],
                )

                # Model should be prefixed with openai/
                call_kwargs = mock_acompletion.call_args[1]
                assert call_kwargs["model"] == "openai/test-model"
                assert call_kwargs["api_base"] == "https://custom.api"


class TestIdentifyParallel:
    """Tests for parallel identify processing."""

    @pytest.mark.asyncio
    async def test_identify_parallel_ordering(self):
        """Test that chunks are returned in correct order."""
        from ontoagain.identify import identify_parallel

        # Mock call_llm_async to return chunk number in result
        async def mock_llm_async(model, messages):
            # Simulate varying response times
            await asyncio.sleep(0.01)
            return f'<C q="test">chunk</C>'

        with patch("ontoagain.identify.call_llm_async", side_effect=mock_llm_async):
            chunks = ["chunk1", "chunk2", "chunk3"]
            results = await identify_parallel(
                chunks=chunks,
                model="test-model",
                ontology_info="",
                max_concurrent=2,
                verbose=False,
            )

            # Should have same number of results as chunks
            assert len(results) == 3

    @pytest.mark.asyncio
    async def test_identify_parallel_concurrency_limit(self):
        """Test that concurrency is limited by semaphore."""
        from ontoagain.identify import identify_parallel

        concurrent_count = 0
        max_concurrent_seen = 0

        async def mock_llm_async(model, messages):
            nonlocal concurrent_count, max_concurrent_seen
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return '<C q="test">result</C>'

        with patch("ontoagain.identify.call_llm_async", side_effect=mock_llm_async):
            chunks = ["c1", "c2", "c3", "c4", "c5"]
            await identify_parallel(
                chunks=chunks,
                model="test-model",
                ontology_info="",
                max_concurrent=2,
                verbose=False,
            )

            # Should never exceed max_concurrent
            assert max_concurrent_seen <= 2


class TestDisambiguateParallel:
    """Tests for parallel disambiguate processing."""

    @pytest.mark.asyncio
    async def test_disambiguate_parallel_merges_results(self):
        """Test that results from all batches are merged."""
        from ontoagain.disambiguate import disambiguate_parallel
        from ontoagain.models import Concept, OntologyMatch

        concepts = [
            Concept(text="p53", context="p53; TP53", start=0, end=3),
            Concept(text="apoptosis", context="apoptosis; PCD", start=10, end=19),
        ]
        all_candidates = [
            [{"id": "PR:001", "ontology": "PR", "label": "p53"}],
            [{"id": "GO:001", "ontology": "GO", "label": "apoptosis"}],
        ]
        clusters = [[0], [1]]  # Each concept in its own batch

        async def mock_batch_async(concepts, candidates, indices, model, batch_num, total, metadata):
            # Return mock matches for each index
            results = {}
            for idx in indices:
                results[idx] = [OntologyMatch(ontology="TEST", id=f"TEST:{idx}", label="test")]
            return batch_num, results

        with patch("ontoagain.disambiguate.disambiguate_batch_async", side_effect=mock_batch_async):
            results = await disambiguate_parallel(
                clusters=clusters,
                concepts=concepts,
                all_candidates=all_candidates,
                model="test-model",
                max_concurrent=2,
                verbose=False,
            )

            # Should have results for both concepts
            assert 0 in results
            assert 1 in results
            assert len(results[0]) == 1
            assert len(results[1]) == 1
