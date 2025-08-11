#!/usr/bin/env python3
"""
RAG Evaluation System for Instagram Content
Uses RAGAS framework to evaluate the quality of retrieval and generation
"""

import json
import os
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd
from datetime import datetime

# RAGAS imports
from ragas import SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy, 
    ContextPrecision,
    ContextRecall,
    AspectCritic
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Local imports
from claude_rag_engine import ClaudeRAGEngine
from local_mistral_synthesizer import LocalMistralSynthesizer


@dataclass
class EvaluationSample:
    """Data structure for evaluation samples"""
    question: str
    expected_answer: Optional[str] = None
    expected_sources: Optional[List[str]] = None
    category: str = "general"
    difficulty: str = "medium"  # easy, medium, hard


@dataclass
class EvaluationResult:
    """Data structure for evaluation results"""
    question: str
    generated_answer: str
    retrieved_contexts: List[str]
    expected_answer: Optional[str]
    faithfulness_score: float
    answer_relevancy_score: float
    context_precision_score: float
    context_recall_score: Optional[float]
    citation_accuracy_score: float
    response_time: float
    category: str
    difficulty: str
    timestamp: str


class InstagramRAGEvaluator:
    """
    Comprehensive RAG evaluation system for Instagram content using RAGAS framework
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the RAG evaluator
        
        Args:
            openai_api_key: OpenAI API key for RAGAS metrics (uses GPT-4 for evaluation)
        """
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            print("⚠️ Warning: No OpenAI API key provided. Some metrics may not work.")
        
        # Initialize RAG system with proper path
        enhanced_data_path = "enhanced_processed"  # Default path
        self.rag_engine = ClaudeRAGEngine(
            enhanced_data_path=enhanced_data_path,
            use_vector_search=True,
            use_local_synthesis=True  # Use Mistral for local synthesis
        )
        self.local_synthesizer = LocalMistralSynthesizer()
        
        # Initialize RAGAS metrics
        self._initialize_metrics()
        
        # Load test datasets
        self.test_samples = self._create_test_dataset()
        
    def _initialize_metrics(self):
        """Initialize RAGAS evaluation metrics"""
        if self.openai_api_key:
            # Use GPT-4 as the evaluator LLM
            evaluator_llm = LangchainLLMWrapper(
                ChatOpenAI(
                    model="gpt-4", 
                    api_key=self.openai_api_key,
                    temperature=0.0
                )
            )
            
            # Initialize embeddings for RAGAS
            evaluator_embeddings = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(
                    api_key=self.openai_api_key,
                    model="text-embedding-ada-002"
                )
            )
            
            # Initialize core RAGAS metrics with embeddings
            self.faithfulness = Faithfulness(llm=evaluator_llm)
            self.answer_relevancy = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings) 
            self.context_precision = ContextPrecision(llm=evaluator_llm)
            self.context_recall = ContextRecall(llm=evaluator_llm)
            
            # Custom citation accuracy metric
            self.citation_accuracy = AspectCritic(
                name="citation_accuracy",
                llm=evaluator_llm,
                definition="Evaluate if the answer properly cites sources using [edhonour_DOCID] format for each factual claim."
            )
        else:
            print("⚠️ RAGAS metrics disabled - no OpenAI API key")
            self.faithfulness = None
            self.answer_relevancy = None
            self.context_precision = None
            self.context_recall = None
            self.citation_accuracy = None
    
    def _create_test_dataset(self) -> List[EvaluationSample]:
        """Create comprehensive test dataset for Instagram RAG evaluation"""
        
        test_samples = [
            # Proxmox/Virtualization Questions
            EvaluationSample(
                question="What are the GPU advantages of Proxmox Server according to Ed?",
                expected_sources=["edhonour_DLWBp_ctC7l"],
                category="proxmox",
                difficulty="easy"
            ),
            EvaluationSample(
                question="How does Ed recommend managing GPU pass-through in Proxmox?",
                expected_sources=["edhonour_DLWBp_ctC7l"], 
                category="proxmox",
                difficulty="medium"
            ),
            EvaluationSample(
                question="What specific tools does Ed mention for GPU management in virtualization?",
                category="proxmox",
                difficulty="hard"
            ),
            
            # AI/ML Questions  
            EvaluationSample(
                question="What does Ed say about fine-tuning language models?",
                category="ai_ml",
                difficulty="medium"
            ),
            EvaluationSample(
                question="Which AI frameworks does Ed recommend for beginners?",
                category="ai_ml", 
                difficulty="easy"
            ),
            EvaluationSample(
                question="What are Ed's thoughts on local vs cloud AI model deployment?",
                category="ai_ml",
                difficulty="hard"
            ),
            
            # Technical Infrastructure
            EvaluationSample(
                question="What server hardware does Ed recommend for home labs?",
                category="infrastructure",
                difficulty="medium"
            ),
            EvaluationSample(
                question="How does Ed suggest setting up a development environment?",
                category="infrastructure",
                difficulty="easy"
            ),
            
            # Specific Technical Details
            EvaluationSample(
                question="What specific model names or specifications does Ed mention?",
                category="technical_details",
                difficulty="hard"
            ),
            EvaluationSample(
                question="What pricing information does Ed share about different tools?",
                category="technical_details", 
                difficulty="medium"
            ),
            
            # Edge Cases & Limitations
            EvaluationSample(
                question="What does Ed think about quantum computing?",
                category="edge_case",
                difficulty="easy",
                expected_answer="Ed doesn't discuss quantum computing in his content"
            ),
            EvaluationSample(
                question="How should I format my resume for tech jobs?",
                category="edge_case", 
                difficulty="easy",
                expected_answer="Ed doesn't discuss resume formatting in his content"
            )
        ]
        
        return test_samples
    
    async def evaluate_single_sample(self, sample: EvaluationSample) -> EvaluationResult:
        """Evaluate a single test sample"""
        print(f"🧪 Evaluating: {sample.question[:50]}...")
        
        start_time = datetime.now()
        
        # Get response from RAG system
        try:
            result = self.rag_engine.extract_smart_answer(sample.question)
            generated_answer = result.primary_answer
            
            # Get context from the documents used in this specific query
            retrieved_contexts = []
            for doc_id in result.documents_used[:10]:  # Limit to 10 docs
                # Find the document content from the engine's loaded documents
                for doc in self.rag_engine.documents:
                    if doc.get('document_id') == doc_id:
                        retrieved_contexts.append(doc.get('content_text', ''))
                        break
            
        except Exception as e:
            print(f"❌ RAG system error: {e}")
            generated_answer = "Error generating response"
            retrieved_contexts = []
        
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate with RAGAS metrics
        evaluation_scores = await self._evaluate_with_ragas(
            question=sample.question,
            answer=generated_answer,
            contexts=retrieved_contexts,
            expected_answer=sample.expected_answer
        )
        
        # Create evaluation result
        result = EvaluationResult(
            question=sample.question,
            generated_answer=generated_answer,
            retrieved_contexts=retrieved_contexts,
            expected_answer=sample.expected_answer,
            faithfulness_score=evaluation_scores.get('faithfulness', 0.0),
            answer_relevancy_score=evaluation_scores.get('answer_relevancy', 0.0), 
            context_precision_score=evaluation_scores.get('context_precision', 0.0),
            context_recall_score=evaluation_scores.get('context_recall', None),
            citation_accuracy_score=evaluation_scores.get('citation_accuracy', 0.0),
            response_time=response_time,
            category=sample.category,
            difficulty=sample.difficulty,
            timestamp=datetime.now().isoformat()
        )
        
        return result
    
    async def _evaluate_with_ragas(
        self, 
        question: str, 
        answer: str, 
        contexts: List[str],
        expected_answer: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate using RAGAS metrics"""
        
        if not self.faithfulness:
            print("⚠️ Skipping RAGAS evaluation - no OpenAI API key")
            return {
                'faithfulness': 0.0,
                'answer_relevancy': 0.0, 
                'context_precision': 0.0,
                'context_recall': 0.0 if expected_answer else None,
                'citation_accuracy': self._manual_citation_accuracy(answer)
            }
        
        try:
            # Create RAGAS sample
            sample_data = {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts
            }
            
            if expected_answer:
                sample_data["reference"] = expected_answer
            
            sample = SingleTurnSample(**sample_data)
            
            # Evaluate metrics
            scores = {}
            
            # Faithfulness (factual accuracy)
            scores['faithfulness'] = await self.faithfulness.single_turn_ascore(sample)
            
            # Answer Relevancy  
            scores['answer_relevancy'] = await self.answer_relevancy.single_turn_ascore(sample)
            
            # Context Precision
            scores['context_precision'] = await self.context_precision.single_turn_ascore(sample)
            
            # Context Recall (only if expected answer provided)
            if expected_answer and self.context_recall:
                scores['context_recall'] = await self.context_recall.single_turn_ascore(sample)
            else:
                scores['context_recall'] = None
                
            # Citation Accuracy
            scores['citation_accuracy'] = await self.citation_accuracy.single_turn_ascore(sample)
            
            return scores
            
        except Exception as e:
            print(f"⚠️ RAGAS evaluation error: {e}")
            return {
                'faithfulness': 0.0,
                'answer_relevancy': 0.0,
                'context_precision': 0.0, 
                'context_recall': 0.0 if expected_answer else None,
                'citation_accuracy': self._manual_citation_accuracy(answer)
            }
    
    def _manual_citation_accuracy(self, answer: str) -> float:
        """Manual citation accuracy calculation (fallback)"""
        if not answer:
            return 0.0
            
        # Count sentences and citations
        sentences = len([s for s in answer.split('.') if s.strip()])
        citations = answer.count('[edhonour_')
        
        if sentences == 0:
            return 0.0
            
        # Citation rate (should be close to 1.0 for good citation coverage)
        citation_rate = min(citations / sentences, 1.0)
        return citation_rate
    
    async def run_full_evaluation(self) -> Dict[str, Any]:
        """Run evaluation on all test samples"""
        print("🚀 Starting comprehensive RAG evaluation...")
        print(f"📊 Test samples: {len(self.test_samples)}")
        
        results = []
        
        for i, sample in enumerate(self.test_samples, 1):
            print(f"\n[{i}/{len(self.test_samples)}] Testing: {sample.category}")
            
            try:
                result = await self.evaluate_single_sample(sample)
                results.append(result)
                
                # Show progress
                print(f"   ✅ Faithfulness: {result.faithfulness_score:.2f}")
                print(f"   ✅ Relevancy: {result.answer_relevancy_score:.2f}")
                print(f"   ✅ Citations: {result.citation_accuracy_score:.2f}")
                print(f"   ⏱️ Time: {result.response_time:.1f}s")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Generate comprehensive report
        report = self._generate_evaluation_report(results)
        
        # Save results
        self._save_results(results, report)
        
        return report
    
    def _generate_evaluation_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        if not results:
            return {"error": "No evaluation results"}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([asdict(r) for r in results])
        
        # Overall metrics
        overall_metrics = {
            'faithfulness': df['faithfulness_score'].mean(),
            'answer_relevancy': df['answer_relevancy_score'].mean(),
            'context_precision': df['context_precision_score'].mean(),
            'citation_accuracy': df['citation_accuracy_score'].mean(),
            'average_response_time': df['response_time'].mean(),
            'total_samples': len(results)
        }
        
        # Context recall (only for samples with expected answers)
        context_recall_scores = df['context_recall_score'].dropna()
        if not context_recall_scores.empty:
            overall_metrics['context_recall'] = context_recall_scores.mean()
        
        # Performance by category
        category_performance = {}
        for category in df['category'].unique():
            cat_df = df[df['category'] == category]
            category_performance[category] = {
                'samples': len(cat_df),
                'faithfulness': cat_df['faithfulness_score'].mean(),
                'answer_relevancy': cat_df['answer_relevancy_score'].mean(),
                'citation_accuracy': cat_df['citation_accuracy_score'].mean(),
                'avg_response_time': cat_df['response_time'].mean()
            }
        
        # Performance by difficulty
        difficulty_performance = {}
        for difficulty in df['difficulty'].unique():
            diff_df = df[df['difficulty'] == difficulty]
            difficulty_performance[difficulty] = {
                'samples': len(diff_df),
                'faithfulness': diff_df['faithfulness_score'].mean(),
                'answer_relevancy': diff_df['answer_relevancy_score'].mean(),
                'citation_accuracy': diff_df['citation_accuracy_score'].mean()
            }
        
        # System strengths and weaknesses
        strengths = []
        weaknesses = []
        
        if overall_metrics['faithfulness'] > 0.8:
            strengths.append("High factual accuracy (faithfulness)")
        else:
            weaknesses.append("Low factual accuracy - may contain hallucinations")
            
        if overall_metrics['answer_relevancy'] > 0.8:
            strengths.append("Highly relevant answers")
        else:
            weaknesses.append("Answers may not fully address questions")
            
        if overall_metrics['citation_accuracy'] > 0.8:
            strengths.append("Excellent source attribution")
        else:
            weaknesses.append("Poor source citation coverage")
        
        if overall_metrics['average_response_time'] < 5.0:
            strengths.append("Fast response times")
        else:
            weaknesses.append("Slow response times")
        
        return {
            'evaluation_date': datetime.now().isoformat(),
            'overall_metrics': overall_metrics,
            'category_performance': category_performance,
            'difficulty_performance': difficulty_performance,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendations': self._generate_recommendations(overall_metrics, weaknesses)
        }
    
    def _generate_recommendations(self, metrics: Dict[str, float], weaknesses: List[str]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if metrics['faithfulness'] < 0.7:
            recommendations.append("Improve document retrieval quality or add fact-checking")
            
        if metrics['answer_relevancy'] < 0.7:
            recommendations.append("Enhance query understanding and response generation")
            
        if metrics['citation_accuracy'] < 0.8:
            recommendations.append("Strengthen citation enforcement in prompts")
            
        if metrics['average_response_time'] > 10.0:
            recommendations.append("Optimize model inference or caching")
            
        if not recommendations:
            recommendations.append("System performing well - consider expanding test coverage")
        
        return recommendations
    
    def _save_results(self, results: List[EvaluationResult], report: Dict[str, Any]):
        """Save evaluation results and report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = f"rag_evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        
        # Save summary report  
        report_file = f"rag_evaluation_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Results saved:")
        print(f"   • {results_file}")
        print(f"   • {report_file}")


# CLI interface for easy testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Instagram RAG Evaluation System")
    parser.add_argument("--openai-key", help="OpenAI API key for RAGAS metrics")
    parser.add_argument("--quick", action="store_true", help="Run quick evaluation (first 3 samples)")
    
    args = parser.parse_args()
    
    async def main():
        evaluator = InstagramRAGEvaluator(openai_api_key=args.openai_key)
        
        if args.quick:
            print("🏃‍♂️ Running quick evaluation (first 3 samples)...")
            evaluator.test_samples = evaluator.test_samples[:3]
        
        report = await evaluator.run_full_evaluation()
        
        print("\n🎯 EVALUATION SUMMARY")
        print("=" * 50)
        print(f"📊 Overall Faithfulness: {report['overall_metrics']['faithfulness']:.2f}")
        print(f"🎯 Answer Relevancy: {report['overall_metrics']['answer_relevancy']:.2f}")
        print(f"📚 Citation Accuracy: {report['overall_metrics']['citation_accuracy']:.2f}")
        print(f"⏱️ Avg Response Time: {report['overall_metrics']['average_response_time']:.1f}s")
        
        print(f"\n✅ Strengths: {', '.join(report['strengths'])}")
        print(f"⚠️ Weaknesses: {', '.join(report['weaknesses'])}")
        print(f"💡 Recommendations: {', '.join(report['recommendations'])}")
    
    asyncio.run(main())