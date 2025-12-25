import frontmatter
import uuid_utils as uuid
import re
import os

class MDProcessor:
    @staticmethod
    def generate_uuid_v7(timestamp=None):
        return str(uuid.uuid7(timestamp=timestamp))

    @staticmethod
    def extract_uuid(content, filename=None):
        """
        Extract UUID from frontmatter or content.
        If filename is a UUID, use it.
        """
        # 1. Try frontmatter
        try:
            post = frontmatter.loads(content)
            if 'id' in post.metadata:
                return str(post.metadata['id']), post.content
            if 'uuid' in post.metadata:
                return str(post.metadata['uuid']), post.content
        except:
            pass

        # 2. Try regex (standard UUID pattern)
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        match = re.search(uuid_pattern, content, re.IGNORECASE)
        if match:
            # We assume the content remains the same unless it's specifically frontmatter
            return match.group(0), content

        # 3. Check filename
        if filename:
            name_without_ext = os.path.splitext(os.path.basename(filename))[0]
            if re.match(uuid_pattern, name_without_ext, re.IGNORECASE):
                return name_without_ext, content

        return None, content

    @staticmethod
    def prepare_metadata(content):
        """Extract basic keywords/metadata if possible from MD structure"""
        meta = {}
        try:
            post = frontmatter.loads(content)
            meta = post.metadata
        except:
            pass
        return meta
