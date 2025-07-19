"""
Import Test Script

Simple script to test if all modules can be imported correctly.

Author: lunazhang
Date: 2023
"""

def test_imports():
    """Test if all core modules can be imported"""
    try:
        # Test core CGCNN imports
        from cgcnn import CrystalGraphConvNet, CIFData
        print("✓ Core CGCNN modules imported successfully")
        
        # Test utility imports
        import utils
        print("✓ Utils module imported successfully")
        
        # Test prediction functionality
        import predict
        print("✓ Prediction module imported successfully")
        
        # Test training functionality
        import main
        print("✓ Training module imported successfully")
        
        # Test active learning (optional due to dependencies)
        try:
            import active_learning
            print("✓ Active learning module imported successfully")
        except ImportError:
            print("⚠ Active learning module requires scikit-optimize")
        
        print("\nCore modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False

if __name__ == "__main__":
    test_imports() 