from app.models.mimic_run import MIMICRun


class MIMICService:

    def run(self, db, user_id, dataset):

        run = MIMICRun(
            dataset_name=dataset,
            result_path="fake/output/path",
            user_id=user_id
        )

        db.add(run)
        db.commit()

        return run