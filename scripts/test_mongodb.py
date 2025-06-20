#!/usr/bin/env python3
"""
Debug MongoDB connection issues
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pymongo import MongoClient

def test_mongodb_connection():
    """Test different MongoDB connection configurations"""
    
    print("🔍 Testing MongoDB connections...")
    
    # Configuration options to try
    configs = [
        {
            "name": "Default (admin/admin)",
            "host": "localhost",
            "port": 27017,
            "username": "admin",
            "password": "password",
            "authSource": "admin"
        },
        {
            "name": "No auth",
            "host": "localhost", 
            "port": 27017
        },
        {
            "name": "Different auth source",
            "host": "localhost",
            "port": 27017,
            "username": "admin",
            "password": "password",
            "authSource": "trading_bot"
        }
    ]
    
    for config in configs:
        print(f"\n🧪 Testing: {config['name']}")
        
        try:
            if 'username' in config:
                client = MongoClient(
                    host=config['host'],
                    port=config['port'],
                    username=config['username'],
                    password=config['password'],
                    authSource=config.get('authSource', 'admin'),
                    serverSelectionTimeoutMS=5000
                )
            else:
                client = MongoClient(
                    host=config['host'],
                    port=config['port'],
                    serverSelectionTimeoutMS=5000
                )
            
            # Test connection
            client.admin.command('ping')
            
            # List databases
            databases = client.list_database_names()
            print(f"✅ Connection successful!")
            print(f"   Available databases: {databases}")
            
            # Test creating trading_bot database
            db = client.trading_bot
            collection = db.test_collection
            collection.insert_one({"test": "data"})
            collection.delete_one({"test": "data"})
            print(f"✅ Can read/write to trading_bot database")
            
            client.close()
            return config
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            try:
                client.close()
            except:
                pass
    
    return None

def check_docker_mongo():
    """Check the Docker MongoDB container"""
    import subprocess
    
    print("\n🐳 Checking Docker MongoDB container...")
    
    try:
        # Get container info
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=mongo", "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("Docker containers:")
            print(result.stdout)
        else:
            print("❌ Error running docker ps")
            
    except Exception as e:
        print(f"❌ Error checking Docker: {e}")

def main():
    print("🚀 MongoDB Connection Debugger")
    
    # Check Docker container first
    check_docker_mongo()
    
    # Test connections
    working_config = test_mongodb_connection()
    
    if working_config:
        print(f"\n✅ Working configuration found: {working_config['name']}")
        print("\n📝 Update your config.yaml with:")
        print("database:")
        print(f"  host: \"{working_config['host']}\"")
        print(f"  port: {working_config['port']}")
        if 'username' in working_config:
            print(f"  username: \"{working_config['username']}\"")
            print(f"  password: \"{working_config['password']}\"")
            print(f"  auth_source: \"{working_config.get('authSource', 'admin')}\"")
        print("  database_name: \"trading_bot\"")
    else:
        print("\n❌ No working MongoDB configuration found")
        print("\n🔧 Possible solutions:")
        print("1. Check if MongoDB container is running: docker ps")
        print("2. Check MongoDB logs: docker logs cka_simulator_mongodb")
        print("3. Try connecting without authentication")
        print("4. Create the trading_bot database manually")

if __name__ == "__main__":
    main()