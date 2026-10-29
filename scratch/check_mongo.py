import pymongo

def main():
    try:
        client = pymongo.MongoClient('mongodb://localhost:27017', serverSelectionTimeoutMS=2000)
        dbs = client.list_database_names()
        print(f"MongoDB Connected Successfully! Active Databases: {dbs}")
        if 'deepguard_db' in dbs:
            db = client['deepguard_db']
            users = list(db.users.find({}, {'_id': 0, 'password_hash': 0}))
            print(f"Registered Users in MongoDB (deepguard_db.users): {len(users)}")
            for u in users:
                print(f" - Email: {u.get('email')}, Name: {u.get('name')}, UserID: {u.get('user_id')}")
        else:
            print("deepguard_db not yet created in MongoDB. Registers will auto-create it upon first registration.")
    except Exception as e:
        print(f"MongoDB Service Offline on localhost:27017 ({e}). DeepGuard is using its built-in In-Memory Database Fallback.")

if __name__ == "__main__":
    main()
