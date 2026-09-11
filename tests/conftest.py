"""
Shared test fixtures and configuration.
"""

import pytest


@pytest.fixture
def sample_article_text_en() -> str:
    """Sample English legal text with articles for testing."""
    return """
BOOK 4 OBLIGATIONS

CHAPTER 1 GENERAL PROVISIONS

Article 1. This Book shall govern obligations arising from contracts,
torts, unjust enrichment, and management of affairs.

Article 2. An obligation is a legal relationship between specific
persons under which one party (the obligor) owes the other party
(the obligee) a specific performance.

CHAPTER 2 FORMATION OF CONTRACTS

Article 315. A contract shall be formed when an offer and an
acceptance thereof are made between the parties and their contents
are in agreement.

Article 316. An offer shall be a manifestation of intention to enter
into a contract made to another party, which contains the essential
elements of the contract with sufficient definiteness.
"""


@pytest.fixture
def sample_article_text_kh() -> str:
    """Sample Khmer legal text with articles for testing."""
    return """
គន្ថី ៤ កាតព្វកិច្ច

ជំពូក ១ បទប្បញ្ញត្តិទូទៅ

មាត្រា ១។ គន្ថីនេះ គ្រប់គ្រងកាតព្វកិច្ចដែលកើតឡើងពីកិច្ចសន្យា

មាត្រា ២។ កាតព្វកិច្ចគឺជាទំនាក់ទំនងផ្លូវច្បាប់រវាងបុគ្គលជាក់លាក់
"""


@pytest.fixture
def sample_extracted_text_kh() -> str:
    """Khmer text in the layout produced by LimonPdfExtractor (one heading/paragraph per line)."""
    return """គន្ថីទី ១ បទប្បញ្ញត្តិទូទៅ
ជំពូកទោល បទប្បញ្ញត្តិទូទៅ
មាត្រា ១.- ច្បាប់ទូទៅនៃនីតិឯកជន
ក្រមនេះ បញ្ញត្តអំពីគោលការណ៍ទូទៅ។ សូមមើលមាត្រា ៣៣៦ នៃក្រមនេះ ។
មាត្រា ២.- គោលការណ៍ជាមូលដ្ឋាន
១-សិទ្ធិឯកជនត្រូវគោរព ។
២-មិនអនុញ្ញាតឱ្យរំលោភសិទ្ធិ ។
គន្ថីទី ២ បុគ្គល
ជំពូកទី ១ បុគ្គលរូបវន្ត
ផ្នែកទី ១ សមត្ថភាពសិទ្ធិ
មាត្រា ៣.- គោលការណ៍ស្វ័យភាពនៃបុគ្គលឯកជន
ក្រមនេះ គោរពនូវសេរីភាពនៃឆន្ទៈរបស់បុគ្គល ។
គន្ថីទី ៣ កាតព្វកិច្ច
ជំពូកទោល បទប្បញ្ញត្តិទូទៅ
មាត្រា ៤.- វិសាលភាព
គន្ថីទី ១ (បទប្បញ្ញត្តិទូទៅ) នៃក្រមនេះ ត្រូវយកមកអនុវត្ត ។
មាតិកាពិស្តារ
មាត្រា ១.- ច្បាប់ទូទៅនៃនីតិឯកជន
មាត្រា ២.- គោលការណ៍ជាមូលដ្ឋាន
"""
