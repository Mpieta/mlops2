# outputs.tf

output "bucket_arns" {
   value = {
      "${var.regions[0]}" = aws_s3_bucket.s3_us_east_1.arn,
	"${var.regions[1]}" = aws_s3_bucket.s3_us_west_2.arn
      # define rest keys in outputs, each for the regions you specified
   }
}

output "bucket_regions" {
   value = {
      "${aws_s3_bucket.s3_us_east_1.id}"     = var.regions[0],
	"${aws_s3_bucket.s3_us_west_2.id}" = var.regions[1]
      # define rest keys in outputs, each for the regions you specified
   }
}