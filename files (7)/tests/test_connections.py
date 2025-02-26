import asyncio
import logging
from core.orchestrator import SuperIntelligenceOrchestrator
from utils.connection_validator import ConnectionValidator

async def test_system_connections():
    """Test all system connections"""
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create validator
    validator = ConnectionValidator()
    
    # Check module connections
    logging.info("Checking module connections...")
    module_results = validator.validate_all_connections()
    
    if not all(module_results.values()):
        failed_modules = [
            module for module, status in module_results.items() 
            if not status
        ]
        logging.error(f"Failed module connections: {failed_modules}")
        return False
        
    # Check file structure
    logging.info("Checking file structure...")
    structure_results = validator.check_file_structure()
    
    if not all(structure_results.values()):
        missing_paths = [
            path for path, exists in structure_results.items() 
            if not exists
        ]
        logging.error(f"Missing directories: {missing_paths}")
        return False
        
    # Initialize orchestrator
    logging.info("Initializing orchestrator...")
    orchestrator = SuperIntelligenceOrchestrator()
    
    # Test system initialization
    if not await orchestrator.initialize_system():
        logging.error("System initialization failed")
        return False
        
    # Get system status
    status = await orchestrator.get_system_status()
    logging.info(f"System Status: {status}")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_system_connections())